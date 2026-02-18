import json

from datasets import concatenate_datasets, load_dataset, Features, Value, Sequence
import argparse

import base64
import zlib
import pickle
LCB_SYSTEM_MESSAGE_GENERIC = "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."

LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."

LCB_FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."

def fetch_live_code_bench_system_prompt(prompt: str, starter_code: str | None = None):
    # https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/prompts/code_generation.py
    prompt = LCB_SYSTEM_MESSAGE_GENERIC + "\n\n" + prompt
    if starter_code:
        prompt += f"### Format: {LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {LCB_FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt
def map_fn(split: str):
    def preprocess_fn(example, idx):
        starter_code = example.get("starter_code", "")
        question = fetch_live_code_bench_system_prompt(example["problem"], starter_code if starter_code else None)

        tests_raw = example["tests"]
        # Handle different test formats
        if isinstance(tests_raw, str):
            tests = json.loads(tests_raw)
        else:
            tests = tests_raw
        metadata = example.get("metadata", {})

        # Convert TACO format to standard format
        if isinstance(tests, dict) and "inputs" in tests and "outputs" in tests:
            normalized_tests = []
            for input_val, output_val in zip(tests["inputs"], tests["outputs"], strict=False):
                normalized_tests.append({"input": input_val, "output": output_val, "testtype": "stdin_stdout"})
            tests = normalized_tests

        # Ensure tests is always a list
        if not isinstance(tests, list):
            tests = [tests] if tests else []

        for test in tests:
            if test.get("testtype") == "functional" and metadata.get("func_name") is not None:
                test["metadata"] = {"func_name": str(metadata["func_name"])}
            else:
                test["metadata"] = {"func_name": None}
        if starter_code is None:
            starter_code = ""
        if metadata is None:
            metadata = {}
        test_cases_compressed = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(tests)))).decode("utf-8") 
        datapoint = {
            "data_source": "livecodebench",
            "prompt":[{"role":"user","content":question}],
            "ability": "code_generation",
            "reward_model": {"style": "rule", "ground_truth": test_cases_compressed},
            "extra_info":{ "split": split, "uid": f"deepcoder_{idx}", "index": idx, "starter_code": starter_code, "metadata": json.dumps(metadata)}
        }
        return datapoint
    return preprocess_fn
def prepare_deepcoder_data(train_size: int = None, test_size: int = None, output_dir: str = None, include_codeforces: bool = False, include_lcbv5: bool = False):
    train_dataset = concatenate_datasets([load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="primeintellect", split="train"), load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="taco", split="train"), load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="lcbv5", split="train")])
    if include_codeforces and include_lcbv5:
        test_dataset = concatenate_datasets([load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="codeforces", split="test"), load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="lcbv5", split="test")])
    elif include_codeforces:
        test_dataset = load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="codeforces", split="test")
    elif include_lcbv5:
        test_dataset = load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="lcbv5", split="test")
    else: 
        raise ValueError("At least one of include_codeforces or include_lcbv5 must be True for the test set.")
   
    if train_size:
        train_dataset = train_dataset.select(range(min(train_size, len(train_dataset))))
    if test_size:
        test_dataset = test_dataset.select(range(min(test_size, len(test_dataset))))
    train_dataset = train_dataset.map(function=map_fn("train"), with_indices=True, writer_batch_size=10, num_proc=16,remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(function=map_fn("test"), with_indices=True, writer_batch_size=10, num_proc=16, remove_columns=test_dataset.column_names)
    if output_dir:
        train_dataset.to_parquet(f"{output_dir}/train.parquet")
        test_dataset.to_parquet(f"{output_dir}/test.parquet")
    return train_dataset, test_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_size", type=int, default=None, help="Number of training examples to process")
    parser.add_argument("--test_size", type=int, default=None, help="Number of test examples to process")
    parser.add_argument("--local_dir", type=str, default=None, help="Directory to save processed datasets")
    parser.add_argument("--exclude_codeforces", action="store_true", help="Exclude Codeforces from test set")
    parser.add_argument("--exclude_lcbv5", action="store_true", help="Exclude LCBv5 from test set")
    args = parser.parse_args()
    train_dataset, test_dataset = prepare_deepcoder_data(train_size=args.train_size, test_size=args.test_size, output_dir=args.local_dir, include_codeforces=not args.exclude_codeforces, include_lcbv5=not args.exclude_lcbv5)
    print(f"  - Train dataset: {len(train_dataset)} examples")
    print(f"  - Test dataset: {len(test_dataset)} examples")
    print(train_dataset[0])