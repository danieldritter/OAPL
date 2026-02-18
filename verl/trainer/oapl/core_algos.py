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
import torch

import verl.utils.torch_functional as verl_F


def compute_policy_loss(
    response_mask: torch.Tensor,
    ref_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    values: torch.Tensor,
    token_level_rewards: torch.Tensor,
    beta2: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    forward_kl = ref_log_prob - log_prob
    forward_kl = verl_F.masked_mean(forward_kl, response_mask)
    reverse_kl = log_prob - ref_log_prob
    is_ratio = torch.exp(log_prob - ref_log_prob)
    reverse_kl = verl_F.masked_mean(is_ratio * reverse_kl, response_mask)
    ref_log_prob_mask = (ref_log_prob * response_mask).sum(dim=-1)
    log_prob_mask = (log_prob * response_mask).sum(dim=-1)
    # gather reward
    reward = (token_level_rewards * response_mask).sum(dim=-1)
    pos_inds = reward > 0
    neg_inds = ~pos_inds
    # compute loss
    advantage = reward - values

    log_prob_ratio = log_prob_mask - ref_log_prob_mask
    residuals = beta2 * log_prob_ratio - advantage
    res_high_inds = residuals > 0
    res_low_inds = ~res_high_inds
    pos_residuals = residuals[pos_inds]
    neg_residuals = residuals[neg_inds]
    res_high_residuals = residuals[res_high_inds]
    res_low_residuals = residuals[res_low_inds]
    loss = residuals**2
    pos_advantage = advantage[pos_inds]
    neg_advantage = advantage[neg_inds]
    final_loss = loss.mean()
    return final_loss, loss[pos_inds], loss[neg_inds], advantage, pos_advantage, neg_advantage, residuals, pos_residuals, neg_residuals, res_high_residuals, res_low_residuals
