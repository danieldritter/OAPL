#!/usr/bin/env python3
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
"""
Parallel version of generate_offline_math_dataset.py using Ray for distributed generation.
Generates offline training dataset from a pretrained model using data parallelism across GPUs.
"""

import os
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List

import transformers 
import datasets
import hydra
import numpy as np
import ray
from omegaconf import OmegaConf
from tqdm import tqdm



from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.device import is_cuda_available
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed
from verl.utils.reward_score.math_verify import compute_score as math_compute_score
from verl.utils.reward_score.rllm_code import compute_score_lcb as code_compute_score
def format_prompt(problem: str, tokenizer: transformers.PreTrainedTokenizer, template_tokenizer: transformers.PreTrainedTokenizer, instruction_following_suffix: str, include_chat_template: bool = True) -> str:
    """Format problem into chat template."""
    prompt = problem + " " + instruction_following_suffix
    if not include_chat_template:
        return prompt
    formatted_prompt = template_tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        max_length=tokenizer.model_max_length,
    )
    assert isinstance(formatted_prompt, str)
    return formatted_prompt


def compute_rewards(responses: List[str], ground_truth: str, score_func) -> List[float]:
    """Compute reward scores for a list of responses."""
    rewards = []
    for response in responses:
        reward = score_func(response, ground_truth)
        if type(reward) is tuple:
            reward = reward[0]
        rewards.append(float(reward))
    return rewards

def extract_solution(solution_str: str) -> str:
    """Extract solution from boxed format."""
    return remove_boxed(last_boxed_only_string(solution_str))
def deduplicate(dataset):
    print(f"Original dataset len: {len(dataset)}")
    prompt_indices = {}
    prompt_response_rewards = {}
    for i,row in tqdm(enumerate(dataset), total=len(dataset), desc="Deduplicating dataset"):
        prompt_str = row['prompt'][0]['content']
        if prompt_str not in prompt_indices:
            prompt_indices[prompt_str] = i
            prompt_response_rewards[prompt_str] = row['response_rewards']
        else:
            assert tuple(row['response_rewards']) == tuple(prompt_response_rewards[prompt_str]), f"Different response_rewards found for the same prompt: {prompt_str}"
    deduplicated_indices = list(prompt_indices.values())
    deduplicated_dataset = dataset.select(deduplicated_indices)
    print(f"Deduplicated dataset len: {len(deduplicated_dataset)}")
    return deduplicated_dataset
@hydra.main(config_path="config", config_name="offline_generation", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    print(
        f"Starting offline dataset generation with model: {config.model.path}, dataset: {config.data.dataset_path}, output_dir: {config.data.output_dir}"
    )
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true"}},
            num_cpus=config.ray_init.num_cpus,
            _temp_dir=config.ray_init.get("temp_dir", "/tmp/ray_data_gen"),
            address=config.ray_init.get("address", None),
        )
    print(ray.cluster_resources())
    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    if config.data.ability == "math":
        compute_score = math_compute_score
    elif config.data.ability == "code":
        compute_score = code_compute_score
    else:
        raise ValueError(f"Unknown ability: {config.data.ability}")

    local_path = copy_to_local(config.model.path)
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    if config.data.get("template_tokenizer", None) is not None:
        template_tokenizer = hf_tokenizer(config.data.template_tokenizer, trust_remote_code=trust_remote_code)
    else:
        template_tokenizer = tokenizer

    assert config.data.num_responses >= 1, "num_responses should always >= 1"

    # Parse filter thresholds
    filter_correct_thresholds = [int(x) for x in config.data.filter_correct_thresholds.split(",")]
    filter_incorrect_thresholds = [int(x) for x in config.data.filter_incorrect_thresholds.split(",")]
    assert len(filter_correct_thresholds) == len(filter_incorrect_thresholds), "filter_correct_thresholds and filter_incorrect_thresholds must have the same length"

    # Load base dataset
    print("Loading base dataset...")
    dataset = datasets.load_dataset("parquet", data_files=config.data.dataset_path)["train"]
    split = config.data.dataset_path.split("/")[-1].split(".")[0]  # e.g., train.parquet -> train
    if "response" in dataset.column_names:
        print("Deduplicating dataset based on prompts...")
        dataset = deduplicate(dataset)
    if config.data.num_samples is not None and config.data.num_samples < len(dataset):
        print(f"Taking random subset of {config.data.num_samples} samples from each split...")
        dataset = dataset.shuffle(seed=config.data.seed).select(range(config.data.num_samples))

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize Ray workers for distributed generation
    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes, use_gpu=True,max_colocate_count=1)
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init, device_name="cuda" if is_cuda_available else "npu")
    wg.init_model()

    # Prepare prompts
    print("Preparing prompts...")
    chat_lst = [prompt for prompt in dataset["prompt"]]
    ground_truths = [reward_model["ground_truth"] for reward_model in dataset["reward_model"]]


    total_samples = len(dataset)
    config_batch_size = config.data.batch_size
    num_batch = -(-total_samples // config_batch_size)
    output_lst = [[] for _ in range(config.data.num_responses)]

    for batch_idx in tqdm(range(num_batch)):
        print(f"[{batch_idx + 1}/{num_batch}] Start to process.")
        batch_chat_lst = chat_lst[batch_idx * config_batch_size : (batch_idx + 1) * config_batch_size]
        templated_input = template_tokenizer.apply_chat_template(batch_chat_lst, add_generation_prompt=True, tokenize=False)
        inputs = tokenizer(templated_input, padding=True, truncation=True, max_length=config.rollout.prompt_length, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        position_ids = compute_position_id_with_mask(attention_mask)
        batch_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}

        data = DataProto.from_dict(batch_dict)
        # Repeat prompts to generate all responses in parallel (like validation does)
        data_repeated = data.repeat(repeat_times=config.data.num_responses, interleave=True)
        data_repeated_padded, pad_size = pad_dataproto_to_divisor(data_repeated, wg.world_size)

        # Generate all responses in a single call
        print(f"[{batch_idx + 1}/{num_batch}] Start to generate {config.data.num_responses} responses in parallel.")
        output_padded = wg.generate_sequences(data_repeated_padded)
        output = unpad_dataproto(output_padded, pad_size=pad_size)

        # Extract all responses
        output_texts = []
        for i in range(len(output)):
            data_item = output[i]
            prompt_length = data_item.batch["prompts"].shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = data_item.batch["responses"][:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            output_texts.append(response_str)

        # Reorganize: output_texts is interleaved [p0_r0, p0_r1, ..., p1_r0, p1_r1, ...]
        # We need to group by response number for output_lst
        for n_sample in range(config.data.num_responses):
            output_lst[n_sample].extend(output_texts[n_sample::config.data.num_responses])

    # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
    output_lst = np.array(output_lst, dtype=object)
    output_lst = np.transpose(output_lst, axes=(1, 0)).tolist()
    # Process responses and compute rewards
    print("Computing rewards and filtering...")
    processed_data = {}  # threshold values to list of dicts
    total_reward = 0.0
    for idx, (prompt, ground_truth, responses) in enumerate(zip(chat_lst, ground_truths, output_lst)):
        ground_truth = ground_truths[idx]
        rewards = compute_rewards(responses, ground_truth, compute_score)
        total_reward += sum(rewards) / config.data.num_responses


        # Apply filtering for each threshold combination
        for filter_correct_threshold, filter_incorrect_threshold in zip(filter_correct_thresholds, filter_incorrect_thresholds):
            if (filter_correct_threshold, filter_incorrect_threshold) not in processed_data:
                processed_data[(filter_correct_threshold, filter_incorrect_threshold)] = []

            # Apply filtering
            if config.data.filter_all_correct and sum(rewards) < filter_correct_threshold:
                continue
            if config.data.filter_all_incorrect and sum(rewards) > (config.data.num_responses - filter_incorrect_threshold):
                continue
            # Add each response as a separate data point
            for response, reward in zip(responses, rewards):
                data_point = {
                    "data_source": config.data.dataset_path,
                    "prompt": prompt,
                    "ability": "math",
                    "reward_model": {"style": "rule", "ground_truth": ground_truth},
                    "extra_info": {"split": split, "index": idx},
                    "response": response,
                    "precomputed_reward": reward,
                    "response_rewards": rewards,
                }
                processed_data[(filter_correct_threshold, filter_incorrect_threshold)].append(data_point)

    avg_reward = total_reward / len(dataset)
    print(f"Average reward for {config.data.num_responses} responses per prompt: {avg_reward}")

    # Save datasets for each threshold combination
    for (fc_thresh, fi_thresh), data in processed_data.items():
        output_subdir = os.path.join(config.data.output_dir, f"correct_{fc_thresh}_incorrect_{fi_thresh}")
        Path(output_subdir).mkdir(parents=True, exist_ok=True)
        print(f"Saving dataset with filter_correct_threshold={fc_thresh} and filter_incorrect_threshold={fi_thresh}...")
        save_dataset(data, output_subdir, split)

    print(f"\nDataset generation complete! Output saved to {config.data.output_dir}")


def save_dataset(data: List[Dict[str, Any]], output_path: str, split: str):
    """Save processed data to parquet format."""
    os.makedirs(output_path, exist_ok=True)

    # Convert to HuggingFace dataset
    dataset = datasets.Dataset.from_list(data)
    # Save as parquet
    output_file = os.path.join(output_path, f"{split}.parquet")
    dataset.to_parquet(output_file)
    print(f"Saved {len(data)} examples to {output_file}")


if __name__ == "__main__":
    main()
