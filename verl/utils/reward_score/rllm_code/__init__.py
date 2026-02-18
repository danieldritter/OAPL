"""
This module contains the RewardCode class, which evaluates code datasets answers
and assigns rewards based on their correctness on unit tests.
"""

import ast
import base64
import json
import multiprocessing
import pickle
import re
import zlib
from multiprocessing import Manager
from typing import Any

from verl.utils.reward_score.rllm_code.livecodebench import run_test as lcb_run_test


def extract_code_from_model(model_response: str):
    """
    Extracts the code from a Markdown-style code block in an LLM output.

    Parameters:
        model_response (str): The text output from the LLM.

    Returns:
        str: The extracted code, or an empty string if no code block is found.
    """
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", model_response, re.DOTALL)
    if not code_blocks:
        return None
    return code_blocks[-1].strip()


def postprocess_lcb_sample(sample):
    sample_inputs = [sample["input"] for sample in sample]
    sample_outputs = [sample["output"] for sample in sample]
    sample_dict = {
        "inputs": sample_inputs,
        "outputs": sample_outputs,
    }

    if sample[0].get("testtype") == "functional":
        metadata = sample[0].get("metadata", {})
        fn_name = metadata.get("func_name", None)
        assert fn_name is not None, f"Function name is not found, check if your LCB data is preprocessed correctly: {metadata}\nSample: {sample}"
        # Fill in the blank
        sample_dict["fn_name"] = fn_name

    sample = {
        "input_output": json.dumps(sample_dict),
    }
    return sample


def _temp_run(sample, generation, debug, result, metadata_list, timeout):
    res, metadata = lcb_run_test(sample, test=generation, debug=debug, timeout=timeout)
    result.append(res)
    metadata_list.append(metadata)


def lcb_check_correctness_v2(sample, generation, timeout=6, debug=False, continuous=False):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""
    assert len(sample) >= 1, "Sample must contain at least one test case"
    sample = postprocess_lcb_sample(sample)

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()

    p = multiprocessing.Process(
        target=_temp_run,
        args=(sample, generation, debug, result, metadata_list, timeout),
    )
    p.start()
    p.join(timeout=(timeout + 1) * len(json.loads(sample["input_output"])["inputs"]) + 5)

    detailed_results = {"all_passed": False, "test_results": [], "total_tests": 0, "passed_tests": 0}

    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample["input_output"])
        # consider that all tests failed
        result.extend([[-1 for i in range(len(in_outs["inputs"]))]])
        detailed_results["total_tests"] = len(in_outs["inputs"])
        detailed_results["test_results"] = [{"input": inp, "expected": out, "passed": False, "error": "global timeout"} for inp, out in zip(in_outs["inputs"], in_outs["outputs"], strict=False)]
        if debug:
            print("global timeout")
        return False, detailed_results

    if not result:
        return False, detailed_results

    # Create detailed test results
    in_outs = json.loads(sample["input_output"])
    detailed_results["total_tests"] = len(result[0])
    detailed_results["test_results"] = [
        {"input": inp, "expected": out, "passed": res == True, "error": metadata_list[0].get("error", None), "error_message": metadata_list[0].get("error_message", None), "output": metadata_list[0].get("output", None)}
        for inp, out, res in zip(in_outs["inputs"], in_outs["outputs"], result[0], strict=False)
    ]
    detailed_results["passed_tests"] = sum(1 for r in result[0] if r == True)
    detailed_results["all_passed"] = all(r == True for r in result[0])
    if not continuous:
        return all(x == True for x in result[0]), detailed_results
    else:
        # Continuous: return True if at least one test passed
        pass_frac = detailed_results["passed_tests"] / detailed_results["total_tests"]
        return pass_frac, detailed_results


def taco_to_lcb_format(tests):
    """
    Given a dictionary with keys "inputs" and "outputs", returns a list of test cases.
    Each test case is a dictionary with keys "input" and "output". If the lists are unequal,
    missing entries are filled by reusing the first element of the shorter list.

    Args:
        data (dict): A dictionary with keys "inputs" and "outputs", each mapped to a list of strings.

    Returns:
        list of dict: A list where each element is a dict with keys "input" and "output".
    """
    inputs = tests.get("inputs", [])
    outputs = tests.get("outputs", [])

    # Determine the number of test cases to create.
    n = max(len(inputs), len(outputs))

    test_cases = []
    for i in range(n):
        # Use the first element as a fallback if the list is shorter than n.
        inp = inputs[i] if i < len(inputs) else (inputs[0] if inputs else "")
        out = outputs[i] if i < len(outputs) else (outputs[0] if outputs else "")
        out = out[0] if isinstance(out, list) else out
        test_case: dict[str, Any] = {"input": inp, "output": out, "metadata": {}}
        if "fn_name" in tests:
            test_case["testtype"] = "functional"
            test_case["metadata"]["func_name"] = tests["fn_name"]
        test_cases.append(test_case)

    return test_cases


def compute_score_lcb(completion: str, test_cases: dict | str, continuous: bool = False):
    """Evaluate code solutions against ground truth answers

        This function creates a reward function to evaluate code solutions by pass the test_case from groun_truth. It can optionally use a language model
        for more sophisticated answer validation.

        Args:
            data_source: The source/dataset the problem comes from
            llm_solution: The solution string provided by the language model to evaluate
            ground_truth: some tests for this llm_solution
            enable_llm: Whether to enable language model validation for complex cases (default: False)

        Returns:
            tuple: (bool, dict) where:
                - bool: True if the solution passes all the test_case, False otherwise
                - dict: Detailed test results with test cases and pass/fail status

        Example:
                model_response = '''
    import sys
    from itertools import permutations
    def main():
        n,m=map(int, input().split())
        a=sum(list(map(int, input().split())))
        if a+(n-1)*10<=m:
            print(5)
        else:
            print(5)
    if __name__ == "__main__":
        main()
    '''

        print(f"test the code_forces")
        # tests = [ { "input": "3 30\n2 2 1", "output": "5" }, { "input": "3 10\n3 2 1", "output": "5" } ]
        metadata = {
             "tests": tests,
        }
        True, {"all_passed": True, "test_results": [...]}
    """
    if not isinstance(test_cases, dict):
        try:
            test_cases = json.loads(test_cases)
        except Exception:
            test_cases = json.loads(pickle.loads(zlib.decompress(base64.b64decode(test_cases.encode("utf-8")))))
    model_code = extract_code_from_model(completion)
    is_correct, test_details = lcb_check_correctness_v2(test_cases, model_code, continuous=continuous)
    return is_correct, test_details


def compute_score_taco_apps_codecontests(completion: str, test_cases: dict | str, continuous: bool = False):
    if not isinstance(test_cases, dict):
        test_cases = json.loads(test_cases)
    model_code = extract_code_from_model(completion)
    is_correct, test_details = lcb_check_correctness_v2(taco_to_lcb_format(test_cases), model_code, continuous=continuous)
    return is_correct, test_details
