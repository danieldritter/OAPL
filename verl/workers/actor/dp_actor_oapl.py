# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Single Process Actor"""

import logging
import os

import torch

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.oapl import core_algos
from verl.trainer.ppo.core_algos import kl_penalty
from verl.utils.debug.performance import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import rearrange_micro_batches
from verl.workers.actor import DataParallelPPOActor

__all__ = ["DataParallelAPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelOAPLActor(DataParallelPPOActor):
    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "values", "token_level_rewards", "ref_log_prob"]
        if multi_turn:
            select_keys.append("loss_mask")
        # if self.config.use_kl_loss:
        # select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    micro_batch_metrics = {}
                    # Support all hardwares
                    if torch.cuda.is_available():
                        idx = torch.cuda.current_device()  # e.g. 0, 1, …
                        device = torch.device(f"cuda:{idx}")
                    elif torch.backends.mps.is_available():  # for Apple Silicon
                        device = torch.device("mps")
                    else:
                        device = torch.device("cpu")
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(device), **data.non_tensor_batch}
                    else:
                        data = data.to(device)  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    values = data["values"]
                    rewards = data["token_level_rewards"]
                    ref_log_prob = data["ref_log_prob"]
                    entropy_coeff = self.config.entropy_coeff

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)
                    seq_log_prob = (log_prob * response_mask).sum(dim=-1)
                    seq_ref_log_prob = (ref_log_prob * response_mask).sum(dim=-1)
                    pg_loss, pos_loss, neg_loss, advantage, pos_advantage, neg_advantage, residuals, pos_residuals, neg_residuals, high_residuals, low_residuals = core_algos.compute_policy_loss(
                        response_mask=response_mask,
                        ref_log_prob=ref_log_prob,
                        log_prob=log_prob,
                        beta2=self.config.beta2,
                        values=values,
                        token_level_rewards=rewards,
                    )
                    forward_kl = ref_log_prob - log_prob
                    token_forward_kl = verl_F.masked_mean(forward_kl, response_mask)
                    seq_forward_kl = (forward_kl * response_mask).sum(dim=-1).mean()
                    reverse_kl = log_prob - ref_log_prob
                    is_ratio = torch.exp(log_prob - ref_log_prob)
                    seq_is_ratio = torch.exp(seq_log_prob - seq_ref_log_prob)
                    clipped_is = torch.clip(seq_log_prob - seq_ref_log_prob, max=5.0)
                    expected_reward = (torch.exp(clipped_is) * rewards.sum(dim=-1)).mean()
                    micro_batch_metrics["actor/iw_reward"] = expected_reward.detach().item()
                    reverse_kl = verl_F.masked_mean(is_ratio * reverse_kl, response_mask)
                    micro_batch_metrics["actor_aux/forward_kl"] = token_forward_kl.detach().item()
                    micro_batch_metrics["actor/seq_forward_kl"] = seq_forward_kl.detach().item()
                    micro_batch_metrics["actor_aux/reverse_kl"] = reverse_kl.detach().item()
                    micro_batch_metrics["actor/avg_advantage"] = advantage.mean().detach().item()
                    micro_batch_metrics["actor_aux/is_token_ratio_mean"] = verl_F.masked_mean(is_ratio, response_mask).detach().item()
                    micro_batch_metrics["actor_aux/is_token_ratio_max"] = is_ratio[response_mask.bool()].max().detach().item()
                    micro_batch_metrics["actor_aux/is_token_ratio_min"] = is_ratio[response_mask.bool()].min().detach().item()
                    micro_batch_metrics["actor_aux/is_token_ratio_std"] = is_ratio[response_mask.bool()].std().detach().item()
                    micro_batch_metrics["actor/is_seq_ratio_mean"] = seq_is_ratio.mean().detach().item()
                    micro_batch_metrics["actor/is_seq_ratio_max"] = seq_is_ratio.max().detach().item()
                    micro_batch_metrics["actor/is_seq_ratio_min"] = seq_is_ratio.min().detach().item()
                    micro_batch_metrics["actor/is_seq_ratio_std"] = seq_is_ratio.std().detach().item()
                    if not pos_loss.shape[0] == 0:
                        micro_batch_metrics["actor_aux/pos_loss"] = pos_loss.mean().detach().item()
                        micro_batch_metrics["residuals/avg_residuals_pos"] = pos_residuals.mean().detach().item()
                        micro_batch_metrics["residuals/max_residuals_pos"] = pos_residuals.max().detach().item()
                        micro_batch_metrics["residuals/min_residuals_pos"] = pos_residuals.min().detach().item()
                        micro_batch_metrics["actor_aux/pos_advantage"] = pos_advantage.mean().detach().item()
                    if not high_residuals.shape[0] == 0:
                        micro_batch_metrics["residuals/avg_high_residuals"] = high_residuals.mean().detach().item()
                        micro_batch_metrics["residuals/max_high_residuals"] = high_residuals.max().detach().item()
                        micro_batch_metrics["residuals/min_high_residuals"] = high_residuals.min().detach().item()
                        if low_residuals.shape[0] == 0:
                            micro_batch_metrics["residuals/high_residuals_frac"] = 1.0
                        else:
                            micro_batch_metrics["residuals/high_residuals_frac"] = high_residuals.shape[0] / (high_residuals.shape[0] + low_residuals.shape[0])
                    if not neg_loss.shape[0] == 0:
                        micro_batch_metrics["actor_aux/neg_loss"] = neg_loss.mean().detach().item()
                        micro_batch_metrics["residuals/avg_residuals_neg"] = neg_residuals.mean().detach().item()
                        micro_batch_metrics["residuals/max_residuals_neg"] = neg_residuals.max().detach().item()
                        micro_batch_metrics["residuals/min_residuals_neg"] = neg_residuals.min().detach().item()
                        micro_batch_metrics["actor_aux/neg_advantage"] = neg_advantage.mean().detach().item()
                    if not low_residuals.shape[0] == 0:
                        micro_batch_metrics["residuals/avg_low_residuals"] = low_residuals.mean().detach().item()
                        micro_batch_metrics["residuals/max_low_residuals"] = low_residuals.max().detach().item()
                        micro_batch_metrics["residuals/min_low_residuals"] = low_residuals.min().detach().item()
                        if high_residuals.shape[0] == 0:
                            micro_batch_metrics["residuals/low_residuals_frac"] = 1.0
                        else:
                            micro_batch_metrics["residuals/low_residuals_frac"] = low_residuals.shape[0] / (high_residuals.shape[0] + low_residuals.shape[0])
                    micro_batch_metrics["residuals/avg_residuals"] = residuals.mean().detach().item()
                    micro_batch_metrics["residuals/max_residuals"] = residuals.max().detach().item()
                    micro_batch_metrics["residuals/min_residuals"] = residuals.min().detach().item()
                    if entropy_coeff != 0:
                        entropy_loss = verl_F.masked_mean(entropy, response_mask)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = verl_F.masked_mean(kld, response_mask)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item()
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    # loss.backward()
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item()
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                data["actor/pos_loss_std"] = torch.tensor(metrics["actor/pos_loss"]).std().item() if "actor/pos_loss" in metrics else 0.0
                data["actor/neg_loss_std"] = torch.tensor(metrics["actor/neg_loss"]).std().item() if "actor/neg_loss" in metrics else 0.0
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
