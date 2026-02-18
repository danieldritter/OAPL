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
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.

"""

import os
from collections import defaultdict

import hydra
import numpy as np
import pandas as pd
import ray
from tqdm import tqdm

from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.fs import copy_to_local
from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.math_verify import compute_score as math_compute_score


def compute_pass_at_k(n, c, k):
    """
    Compute the pass@k score.
    :param n: number of samples
    :param c: number of correct answers
    :param k: k value
    :return: pass@k score
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


@ray.remote
def process_item(reward_fn, data_source, response_lst, reward_data):
    ground_truth = reward_data["ground_truth"]
    # score_lst = [reward_fn(data_source, r, ground_truth) for r in response_lst]
    score_lst = [reward_fn(r, ground_truth) for r in response_lst]
    return data_source, np.mean(score_lst)


@ray.remote
def process_item_pass_at_k(reward_fn, data_source, response_lst, reward_data):
    ground_truth = reward_data["ground_truth"]
    # score_lst = [reward_fn(data_source, r, ground_truth) for r in response_lst]
    n = len(response_lst)
    c = sum([1 for r in response_lst if reward_fn(r, ground_truth)])
    pass_at_ks = [compute_pass_at_k(n, c, k) for k in range(1, n + 1)]
    return data_source, pass_at_ks


@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def main(config):
    local_path = copy_to_local(config.data.path, use_shm=config.data.get("use_shm", False))
    dataset = pd.read_parquet(local_path)
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]
    total = len(dataset)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=config.ray_init.num_cpus, _temp_dir="/tmp/dritter/eval_ray", address="local")

    # evaluate test_score based on data source
    data_source_reward = defaultdict(list)
    data_source_reward_pass_at_k = defaultdict(list)
    compute_score = get_custom_reward_fn(config)
    if compute_score is None:
        compute_score = default_compute_score
    compute_score = math_compute_score
    # Create remote tasks
    remote_tasks = [process_item.remote(compute_score, data_sources[i], responses[i], reward_model_data[i]) for i in range(total)]
    remote_tasks_pass_at_k = [process_item_pass_at_k.remote(compute_score, data_sources[i], responses[i], reward_model_data[i]) for i in range(total)]

    # Process results as they come in
    with tqdm(total=total) as pbar:
        while len(remote_tasks) > 0:
            # Use ray.wait to get completed tasks
            done_ids, remote_tasks = ray.wait(remote_tasks)
            for result_id in done_ids:
                data_source, score = ray.get(result_id)
                data_source_reward[data_source].append(score)
                pbar.update(1)
    with tqdm(total=total) as pbar_pass_at_k:
        while len(remote_tasks_pass_at_k) > 0:
            done_ids, remote_tasks_pass_at_k = ray.wait(remote_tasks_pass_at_k)
            for result_id in done_ids:
                data_source, pass_at_ks = ray.get(result_id)
                data_source_reward_pass_at_k[data_source].append(pass_at_ks)
                pbar_pass_at_k.update(1)

    metric_dict = {}
    for data_source, rewards in data_source_reward.items():
        metric_dict[f"avg@{len(responses[0])}/{data_source}"] = np.mean(rewards)
    for data_source, rewards in data_source_reward_pass_at_k.items():
        for k in range(1, len(responses[0]) + 1):
            metric_dict[f"pass@{k}/{data_source}"] = np.mean([r[k - 1] for r in rewards])
    print(metric_dict)

    # Save the results to a file
    output_path = config.data.get("output_path", None)
    if output_path:
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        output_df = pd.DataFrame.from_dict(metric_dict, orient="index", columns=["score"])
        output_df.to_csv(output_path, index_label="data_source")
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
