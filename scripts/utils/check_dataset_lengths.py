# script to check the maximum prompt lengths of a dataset (to set max in configs)
import argparse

import datasets
import matplotlib.pyplot as plt
import polars as pl
from tqdm import tqdm

from verl.utils import hf_tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check the maximum prompt lengths of a dataset",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to or name of the model",
    )
    parser.add_argument(
        "--dataset_files",
        nargs="+",
        type=str,
        required=True,
        help="Path to or name of the dataset files",
    )
    parser.add_argument("--prompt_key", type=str, default="prompt")
    parser.add_argument("--response_key", type=str, default="response")
    parser.add_argument("--max_prompt_length", type=int)
    parser.add_argument("--max_response_length", type=int)
    args = parser.parse_args()
    tokenizer = hf_tokenizer(name_or_path=args.model_name_or_path)

    # load datasets
    dataframes = []
    for parquet_file in args.dataset_files:
        df = pl.read_parquet(parquet_file)
        dataframe = datasets.Dataset.from_polars(df)  # better for large files
        dataframes.append(dataframe)
    dataframe = datasets.concatenate_datasets(dataframes)
    num_no_think = 0
    for doc in tqdm(dataframe, desc="Checking for </think> tags"):
        if "</think>" not in doc[args.response_key]:
            num_no_think += 1
    print(f"Number of samples without </think>: {num_no_think} / {len(dataframe)}")
    if "precomputed_reward" in dataframe.column_names:
        avg_reward = sum(dataframe["precomputed_reward"]) / len(dataframe)
        print(f"Average reward across dataset: {avg_reward}")
    orig_len = len(dataframe)
    print(f"Loaded {orig_len} samples from {args.dataset_files}")
    max_prompt_length = -1
    max_response_length = -1
    num_prompts_too_long = 0
    num_responses_too_long = 0
    prompt_lens = []
    response_lens = []
    from concurrent.futures import ThreadPoolExecutor

    def process_doc(doc):
        prompt = tokenizer.apply_chat_template(
            doc[args.prompt_key],
            add_generation_prompt=True,
        )
        if args.response_key in doc:
            response = tokenizer.encode(doc[args.response_key])
        return len(prompt), len(response)

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(process_doc, [doc for doc in dataframe]), total=len(dataframe)))
    prompt_lens, response_lens = zip(*results)
    max_prompt_length = max(prompt_lens)
    max_response_length = max(response_lens)
    num_prompts_too_long = sum(l > args.max_prompt_length for l in prompt_lens)
    num_responses_too_long = sum(l > args.max_response_length for l in response_lens)
    print(f"Maximum prompt length across dataset: {max_prompt_length}")
    print(f"Maximum response length across dataset: {max_response_length}")
    print(f"Number of prompts longer than {args.max_prompt_length}: {num_prompts_too_long}")
    print(f"Number of responses longer than {args.max_response_length}: {num_responses_too_long}")
    print(f"Percentage of prompts longer than {args.max_prompt_length}: {num_prompts_too_long / len(dataframe) * 100:.2f}%")
    print(f"Percentage of responses longer than {args.max_response_length}: {num_responses_too_long / len(dataframe) * 100:.2f}%")
    print(f"Number of samples after filtering prompts: {len(dataframe) - num_prompts_too_long}")
    print(f"Number of samples after filtering responses: {len(dataframe) - num_responses_too_long}")
    print(f"Response lengths greater than max_response_length: {[l for l in response_lens if l > args.max_response_length]}")
    # plot the response lengths histogram
    plt.figure(figsize=(6, 6))
    plt.hist(response_lens, bins=100, color="green", alpha=0.7)
    plt.axvline(args.max_response_length, color="red", linestyle="dashed", linewidth=1)
    plt.title("Response Lengths")
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("response_lengths_histogram.png")
    print("Saved response lengths histogram to response_lengths_histogram.png")
