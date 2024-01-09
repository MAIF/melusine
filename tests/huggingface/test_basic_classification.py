from unittest.mock import patch

import pandas as pd
import pytest

transformers = pytest.importorskip("transformers")
from typing import List

from transformers.pipelines.zero_shot_classification import (
    ZeroShotClassificationPipeline,
)


class MockZeroShotClassificationPipeline(ZeroShotClassificationPipeline):
    def __init__(self, task, model, tokenizer):
        pass

    def __call__(self, sequences, candidate_labels, hypothesis_template):
        if isinstance(sequences, List):
            # Standard return with 2 elements
            return [
                {
                    "sequence": sequences[0],
                    "labels": ["négatif", "positif"],
                    "scores": [0.5, 0.5],
                },
                {
                    "sequence": sequences[1],
                    "labels": ["négatif", "positif"],
                    "scores": [0.5, 0.5],
                },
            ]

        if "gentillesse" in sequences:
            return {
                "sequence": sequences,
                "labels": ["positif", "négatif"],
                "scores": [0.9756866097450256, 0.024313366040587425],
            }
        elif "pas satisfait" in sequences:
            return {
                "sequence": sequences,
                "labels": ["négatif", "positif"],
                "scores": [0.7485730648040771, 0.25142696499824524],
            }
        else:
            return {
                "sequence": sequences,
                "labels": ["négatif", "positif"],
                "scores": [0.5, 0.5],
            }


def test_tutorial001(add_docs_to_pythonpath):
    from docs_src.BasicClassification.tutorial001 import run, transformers_standalone

    with patch(
        "docs_src.BasicClassification.tutorial001.pipeline",
        new=MockZeroShotClassificationPipeline,
    ):
        result = transformers_standalone()
        assert isinstance(result, List)

        df = run()
        assert isinstance(df, pd.DataFrame)
