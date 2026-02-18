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

import logging
from typing import Callable, Optional, Sequence

from math_verify.errors import TimeoutException
from math_verify.grader import verify
from math_verify.parser import ExprExtractionConfig, ExtractionTarget, LatexExtractionConfig, parse
from math_verify.utils import timeout

logger = logging.getLogger(__name__)


def math_metric(
    gold_extraction_target: Sequence[ExtractionTarget] = (ExprExtractionConfig(),),
    pred_extraction_target: Sequence[ExtractionTarget] = (ExprExtractionConfig(),),
    aggregation_function: Callable[[list[float]], float] = max,
    precision: int = 6,
) -> Callable[[list[str], list[str]], tuple[float, Optional[tuple[list[str], list[str]]]]]:
    """Creates a language-aware extractive match metric that extracts answers from the model's output.

    Known issues:
    - If the task is to simplify an expression, the metric might overestimate the accuracy. This is because if the model doesn't output any anchor for the extraction (e.g final answer is..),
        it's possible that the the extracted prediction will be the expression to simplify. Because we do simplifications ourselves, it can thus happen that sympy will correctly simplify the expression,
        thus it will match gold, despite model not doing anything. PRs to fix this are welcome.

    Args:
        language: Language
            The language of the samples.
        gold_extraction_target: Sequence[ExtractionTarget]
            Extraction targets to use for gold answers. Defaults to extracting simple math expressions.
        pred_extraction_target: Sequence[ExtractionTarget]
            Extraction targets to use for predictions. Defaults to extracting simple math expressions.
        aggregation_function: Callable[[list[float]], float]
            Function to aggregate scores when multiple golds/predictions are present. Defaults to max.
        fallback_mode: Literal["no_fallback", "first_match"]
            How to perform extraction. Defaults to "first_match".
            - "no_fallback": Only use first successfully parsed matches
            - "first_match": Use the first successfully parsed match + first match irregardless the parsing success
        precision: int
            Number of decimal places to use when comparing numerical values. Defaults to 6.

    Returns:
        A sample level metric that extracts and compares mathematical expressions.

    """

    @timeout(30)
    def get_str_preds_with_timeout(extracted_predictions: list[list[str]], extracted_golds: list[list[str]]) -> tuple[list[str], list[str]]:
        golds = [str(gold) for golds in extracted_golds for gold in golds]
        predictions = [str(pred) for preds in extracted_predictions for pred in preds]
        return (golds, predictions)

    def sample_level_fn(golds: list[str], predictions: list[str]) -> tuple[float, Optional[tuple[list[str], list[str]]]]:
        extracted_predictions = [parse(pred, pred_extraction_target) for pred in predictions]
        extracted_golds = [parse(gold, gold_extraction_target, parsing_timeout=30) for gold in golds]

        # Assert on empty gold and warn on empty pred
        if any(len(g) == 0 for g in extracted_golds):
            raise ValueError(f"No gold targets found for at least one gold. Gold: {golds}, Pred: {predictions}")

        # if all(len(p) == 0 for p in extracted_predictions):
        # truncated_response = predictions[0][:50] + "..." if len(predictions[0]) > 50 else predictions[0]
        # logger.warning(f"We did not manage to extract a prediction in the correct format. Gold: {golds}, Example Pred: {truncated_response}")

        # We have to use timeout because the sypmy to str conversion can be very slow
        str_preds = None
        try:
            str_preds = get_str_preds_with_timeout(extracted_predictions, extracted_golds)
        except TimeoutException:
            logger.warning("Timeout when adding extracted predictions and golds to specific")

        return (
            aggregation_function([(1.0 if any(verify(gold, pred, precision, timeout_seconds=30) for gold in extracted_golds) else 0.0) for pred in extracted_predictions]),
            str_preds,
        )

    return sample_level_fn


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0) -> bool:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    # ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except Exception:
        pass
    except TimeoutException:
        ret_score = timeout_score

    return ret_score
