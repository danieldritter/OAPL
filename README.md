<h1 align="center">Optimal Advantage-based Policy Optimization with a Lagged Inference Policy (OAPL)</h1>

<div>
<br>

<div align="center">

[![Github](https://img.shields.io/badge/Repo-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/danieldritter/OAPL)
[![arXiv](https://img.shields.io/badge/Paper-red?style=for-the-badge&logo=arXiv&logoColor=white&labelColor)](TBD)
[![Hugging Face Collection](https://img.shields.io/badge/Dataset/Models-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor)](https://huggingface.co/collections/danieldritter/oapl)

</div>

</div>


### **Optimal Advantage-based Policy Optimization with a Lagged Inference Policy**

OAPL is an off-policy algorithm for LLM RL post-training, which uses the closed-form solution to KL-regularized RL to explicitly minimize the divergence between the trainer and inference policies, allowing for effective training with highly off-policy data. This repository contains the code for the offline training/code-generation experiments replicating the performance of [DeepCoder](https://www.together.ai/blog/deepcoder), and is built on [VeRL](https://github.com/verl-project/verl). 


---

## Installation
All experiments used vllm as the inference engine, and the required packages can be installed using the script at
`scripts/utils/install_vllm_sglang_mcore.sh` as below: 
```bash
conda create -n oapl python=3.10
conda activate oapl
USE_MEGATRON=0 USE_SGLANG=0 ./scripts/utils/install_vllm_sglang_mcore.sh
```

## Datasets
Our code model was trained via two rounds of offline training. The datasets for both rounds are available on [huggingface](https://huggingface.co/collections/danieldritter/oapl). To get the livecodebench evaluation set, run `scripts/data/process_datasets.sh`, which will download and process the original deepcoder train and test sets (excluding codeforces from the test set by default).

To generate a new dataset of offline responses from a model (for training or evaluation) in the same format as those available on huggingface, use the scripts at `scripts/data/generate_offline_dataset.sh` or `scripts/data/generate_offline_dataset_multinode.sh`.  

## Training

Scripts for single-node or multi-node training can be found at `scripts/oapl_deepcoder_offline.sh` or `scripts/oapl_deepcoder_offline_multinode.sh`, respectively. The trained models from both rounds of our offline training are also available on [huggingface](https://huggingface.co/collections/danieldritter/oapl).

## Evaluation
To evaluate a model, first generate an offline dataset of responses from the model on the eval dataset (see 'Datasets' above). Then run `scripts/evaluation/offline_dataset_pass_at_k.sh` with that offline dataset to generate a csv with the Pass@k values for various k. To make Pass@k line plots with one or multiple Pass@k curves (e.g. to compare several models), use `scripts/evaluation/make_pass_at_k_plots.sh`, passing in a list of Pass@k files from the previous script. 

## Citing OAPL

```bibtex
@misc{llmscanlearntoreasonfromoffpolicydata,
      title={LLMs Can Learn to Reason From Off-Policy Data}, 
      author={Daniel Ritter and Owen Oertell and Bradley Guo and Jonathan Chang and Kianté Brantley and Wen Sun},
      year={2026},
      eprint={2505.20686},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.20686}, 
}
```
