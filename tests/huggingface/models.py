from unittest.mock import MagicMock, patch

import pytest
import torch

from hugging_face.detectors import DissatisfactionDetector
from hugging_face.models.model import TextClassifier


def return_value(resp, content):
    return content


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.return_value = {"input_ids": [[101, 102]], "attention_mask": [[1, 1]]}
    return tokenizer


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.return_value.logits = torch.tensor([[0.1, 0.9]])  # Simulated logits
    return model


@pytest.fixture
def mock_detector(mock_tokenizer, mock_model):
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        with patch("transformers.AutoModelForSequenceClassification.from_pretrained", return_value=mock_model):
            # Create a TextClassifier instance
            classifier = TextClassifier(
                tokenizer_name_or_path="mock_tokenizer",
                model_name_or_path="mock_model",
                token=None,
            )

            # Create the DissatisfactionDetector using the mock classifier
            detector = DissatisfactionDetector(
                name="dissatisfaction",
                text_column="det_normalized_last_body",
                model_name_or_path="mock_model_path",
                tokenizer_name_or_path="mock_tokenizer_path",
                token=None,
            )
            detector.melusine_model = classifier
            return detector


# Example test using the mock_detector fixture
def test_mock_detector_instantiation(mock_detector):
    assert isinstance(mock_detector, DissatisfactionDetector)
    assert mock_detector.name == "dissatisfaction"
    assert mock_detector.text_column == "det_normalized_last_body"
