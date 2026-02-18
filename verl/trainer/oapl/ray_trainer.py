# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import uuid
from pprint import pprint

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from tqdm import tqdm

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role, apply_kl_penalty, compute_response_mask
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.debug.performance import _timer
from verl.utils.metric import (
    reduce_metrics,
)


def _compute_response_info(batch):
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    if use_critic:
        values = batch.batch["values"]
    else:
        values = torch.zeros_like(sequence_reward)
    metrics = {
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(values).detach().item(),
                "critic/values/max": torch.max(values).detach().item(),
                "critic/values/min": torch.min(values).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    # tracking importance weighted rewards in the offline case
    if "importance_weighted_rewards" in batch.batch:
        iw_sequence_reward = batch.batch["importance_weighted_rewards"]
        metrics.update(
            {
                "critic/iw_rewards/mean": torch.mean(iw_sequence_reward).detach().item(),
                "critic/iw_rewards/max": torch.max(iw_sequence_reward).detach().item(),
                "critic/iw_rewards/min": torch.min(iw_sequence_reward).detach().item(),
                "critic/iw_rewards/std": torch.std(iw_sequence_reward).detach().item(),
                "critic/iw_rewards/sum": torch.sum(iw_sequence_reward).detach().item(),
            }
        )
    return metrics


def add_precomputed_rewards(batch):
    reward_tensor = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
    for i in range(len(batch)):
        data_item = batch[i]
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()

        precomputed_reward = data_item.batch["precomputed_reward"]
        reward_tensor[i, valid_response_length - 1] = precomputed_reward
    return reward_tensor, {}


def compute_log_p_metrics(log_prob: torch.Tensor, ref_log_prob: torch.Tensor, response_mask: torch.Tensor, pos_inds: torch.Tensor) -> dict:
    """Compute average, max, min log probabilities per token for both current and reference policies."""
    # log_prob, ref_log_prob: (bsz, response_length)
    # response_mask: (bsz, response_length)
    metrics = {}
    for prefix in ["probs/all", "probs/pos", "probs/neg"]:
        if prefix == "probs/all":
            log_prob_subset = log_prob
            ref_log_prob_subset = ref_log_prob
            response_mask_subset = response_mask
        elif prefix == "probs/pos":
            log_prob_subset = log_prob[pos_inds]
            ref_log_prob_subset = ref_log_prob[pos_inds]
            response_mask_subset = response_mask[pos_inds]
        else:  # neg
            log_prob_subset = log_prob[~pos_inds]
            ref_log_prob_subset = ref_log_prob[~pos_inds]
            response_mask_subset = response_mask[~pos_inds]
        if log_prob_subset.numel() == 0:
            continue
        metrics[f"{prefix}/token_avg_log_pi"] = verl_F.masked_mean(log_prob_subset, response_mask_subset)
        metrics[f"{prefix}/token_avg_log_pi_ref"] = verl_F.masked_mean(ref_log_prob_subset, response_mask_subset)
        metrics[f"{prefix}/token_max_log_pi"] = (log_prob_subset[response_mask_subset.bool()]).max()
        metrics[f"{prefix}/token_max_log_pi_ref"] = (ref_log_prob_subset[response_mask_subset.bool()]).max()
        metrics[f"{prefix}/token_min_log_pi"] = (log_prob_subset[response_mask_subset.bool()]).min()
        metrics[f"{prefix}/token_min_log_pi_ref"] = (ref_log_prob_subset[response_mask_subset.bool()]).min()
        token_log_ratio = log_prob_subset - ref_log_prob_subset
        metrics[f"{prefix}/token_avg_logratio"] = verl_F.masked_mean(token_log_ratio, response_mask_subset)
        metrics[f"{prefix}/token_max_logratio"] = (token_log_ratio[response_mask_subset.bool()]).max()
        metrics[f"{prefix}/token_min_logratio"] = (token_log_ratio[response_mask_subset.bool()]).min()
        # sequence level metrics
        seq_log_prob = (log_prob_subset * response_mask_subset).sum(dim=-1)  # (bsz,)
        seq_ref_log_prob = (ref_log_prob_subset * response_mask_subset).sum(dim=-1)  # (bsz,)
        seq_log_ratio = seq_log_prob - seq_ref_log_prob
        metrics[f"{prefix}/seq_avg_log_pi"] = seq_log_prob.mean()
        metrics[f"{prefix}/seq_avg_log_pi_ref"] = seq_ref_log_prob.mean()
        metrics[f"{prefix}/seq_max_log_pi"] = seq_log_prob.max()
        metrics[f"{prefix}/seq_max_log_pi_ref"] = seq_ref_log_prob.max()
        metrics[f"{prefix}/seq_min_log_pi"] = seq_log_prob.min()
        metrics[f"{prefix}/seq_min_log_pi_ref"] = seq_ref_log_prob.min()
        metrics[f"{prefix}/seq_avg_logratio"] = seq_log_ratio.mean()
        metrics[f"{prefix}/seq_max_logratio"] = seq_log_ratio.max()
        metrics[f"{prefix}/seq_min_logratio"] = seq_log_ratio.min()
    if "probs/pos/seq_avg_logratio" in metrics and "probs/neg/seq_avg_logratio" in metrics:
        metrics["probs/pos_neg_delta_seq_avg_logratio"] = metrics["probs/pos/seq_avg_logratio"] - metrics["probs/neg/seq_avg_logratio"]
        metrics["probs/pos_neg_delta_token_avg_logratio"] = metrics["probs/pos/token_avg_logratio"] - metrics["probs/neg/token_avg_logratio"]
        metrics["probs/pos_neg_delta_seq_max_logratio"] = metrics["probs/pos/seq_max_logratio"] - metrics["probs/neg/seq_max_logratio"]
        metrics["probs/pos_neg_delta_token_max_logratio"] = metrics["probs/pos/token_max_logratio"] - metrics["probs/neg/token_max_logratio"]
        metrics["probs/pos_neg_delta_seq_min_logratio"] = metrics["probs/pos/seq_min_logratio"] - metrics["probs/neg/seq_min_logratio"]
        metrics["probs/pos_neg_delta_token_min_logratio"] = metrics["probs/pos/token_min_logratio"] - metrics["probs/neg/token_min_logratio"]
    return metrics


class RayOAPLTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        if self.config.data.use_precomputed_values:
            self.use_critic = False
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.workers.rollout.async_server import AsyncLLMServerManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def fit(self):
        """
        The online training loop (original implementation).
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        if self.config.trainer.save_freq == "epoch":
            self.save_freq = len(self.train_dataloader)
        else:
            self.save_freq = self.config.trainer.save_freq
        if self.config.trainer.test_freq == "epoch":
            self.test_freq = len(self.train_dataloader)
        else:
            self.test_freq = self.config.trainer.test_freq

        # torch.use_deterministic_algorithms(True)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch = DataProto.from_single_dict(batch_dict)
                unique_indices_frac = torch.unique(batch.batch["input_ids"], dim=-1).shape[0] / self.config.data.train_batch_size
                metrics.update({"training/unique_problem_indices_frac": unique_indices_frac})
                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    if not self.config.data.run_offline:
                        # generate a batch
                        with _timer("gen", timing_raw):
                            if not self.async_rollout_mode:
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                            else:
                                self.async_rollout_manager.wake_up()
                                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                                self.async_rollout_manager.sleep()
                            timing_raw.update(gen_batch_output.meta_info["timing"])
                            gen_batch_output.meta_info.pop("timing", None)
                            batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                            # repeat to align with repeated responses in rollout
                            batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                            batch = batch.union(gen_batch_output)
                    else:
                        # generating to test for side effects of rollout generation
                        # use offline responses
                        with torch.no_grad():
                            with _timer("offline_gen", timing_raw):
                                meta_info = {"eos_token_id": self.tokenizer.eos_token_id, "pad_token_id": self.tokenizer.pad_token_id}
                                batch.meta_info.update(meta_info)
                                offline_responses = batch.batch.pop("offline_response")
                                offline_response_attention_mask = batch.batch.pop("offline_response_attention_mask")
                                # Create attention mask and position ids for responses
                                # prompt_input_ids = batch.batch.pop("input_ids")
                                # prompt_position_ids = batch.batch.pop("position_ids")
                                prompt_input_ids = gen_batch.batch["input_ids"]
                                prompt_position_ids = gen_batch.batch["position_ids"]
                                delta_position_id = torch.arange(1, offline_responses.size(1) + 1, device=prompt_position_ids.device)
                                delta_position_id = delta_position_id.unsqueeze(0).expand(prompt_input_ids.size(0), -1)
                                response_position_ids = prompt_position_ids[..., -1:] + delta_position_id

                                # Combine prompts and responses
                                combined_input_ids = torch.cat([prompt_input_ids, offline_responses], dim=-1)
                                # prompt_attention_mask = batch.batch.pop("attention_mask")
                                prompt_attention_mask = gen_batch.batch["attention_mask"]
                                combined_attention_mask = torch.cat([prompt_attention_mask, offline_response_attention_mask], dim=-1)
                                # Create position ids for the combined sequence
                                combined_position_ids = torch.cat([prompt_position_ids, response_position_ids], dim=-1)
                                # Create the batch output similar to generate_sequences
                                gen_batch_output = TensorDict(
                                    {"input_ids": combined_input_ids, "attention_mask": combined_attention_mask, "position_ids": combined_position_ids, "responses": offline_responses, "prompts": prompt_input_ids},
                                    batch_size=combined_input_ids.size(0),
                                )
                                gen_batch_output = DataProto(batch=gen_batch_output, non_tensor_batch=gen_batch.non_tensor_batch)
                                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                                batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                                batch = batch.union(gen_batch_output)
                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        if self.config.data.run_offline and "precomputed_reward" in batch.batch:
                            reward_tensor, reward_extra_infos_dict = add_precomputed_rewards(batch)
                        else:
                            # compute reward model score
                            if self.use_rm:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(reward_tensor)

                            if self.config.reward_model.launch_reward_fn_async:
                                future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                            else:
                                reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)
                        # batch.batch["rollout_log_probs"] = batch.batch["old_log_probs"].clone()
                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)
                    # compute values
                    assert not self.use_critic  # No critic in A*PO
                    assert "response_rewards" in batch.batch, "If not using critic, response_rewards must be provided in the batch."
                    response_rewards = batch.batch.pop("response_rewards")
                    beta = self.config.critic.beta1
                    if self.config.critic.convert_to_v_star:
                        values = torch.exp(response_rewards / beta)
                        values = beta * torch.log(values.mean(dim=-1))
                    else:
                        values = torch.mean(response_rewards, dim=-1)
                    batch.batch["values"] = values  # (batch_size,)
                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict = {}
                        if self.config.reward_model.launch_reward_fn_async and not (self.config.data.run_offline and "precomputed_reward" in batch.batch):
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                        if "ref_log_prob" in batch.batch:
                            pos_inds = batch.batch["token_level_rewards"].sum(dim=-1) > 0
                            logp_metrics = compute_log_p_metrics(
                                log_prob=batch.batch["old_log_probs"],
                                ref_log_prob=batch.batch["ref_log_prob"],
                                response_mask=batch.batch["response_mask"],
                                pos_inds=pos_inds,
                            )
                            metrics.update(logp_metrics)
                            metrics["probs/frac_pos"] = pos_inds.float().mean().detach().item()
                            metrics["probs/frac_neg"] = (~pos_inds).float().mean().detach().item()
                        if self.config.data.run_offline:
                            log_p = (batch.batch["old_log_probs"] * batch.batch["response_mask"]).sum(dim=-1)
                            ref_log_p = (batch.batch["ref_log_prob"] * batch.batch["response_mask"]).sum(dim=-1)
                            clipped_log_ratio = torch.clip(log_p - ref_log_p, max=5.0)
                            batch.batch["importance_weighted_rewards"] = batch.batch["token_level_rewards"].sum(dim=-1) * torch.exp(clipped_log_ratio)
                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)

                            scores = batch.batch["token_level_scores"].sum(dim=-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.test_freq > 0 and (is_last_step or self.global_steps % self.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)
                    print("Validation Complete")
                    if self.save_freq > 0 and (is_last_step or self.global_steps % self.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()
                    print("Checkpoint Saved")
                # training metrics
                metrics.update({"training/global_step": self.global_steps, "training/epoch": epoch})
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic or self.config.data.use_precomputed_values))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
