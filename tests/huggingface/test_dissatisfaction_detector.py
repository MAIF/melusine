"""
Unit tests of the DissatisfactionDetector
The model used inside of the detector is mocked in the fixtures tests
"""

from unittest.mock import MagicMock, patch

import pytest
from pandas import DataFrame

from hugging_face.detectors import DissatisfactionDetector
from hugging_face.models.model import TextClassifier


@pytest.mark.usefixtures("mock_detector")
def test_instantiation(mock_detector):
    """Test that the mock detector is instantiated correctly."""
    assert isinstance(mock_detector, DissatisfactionDetector)
    assert mock_detector.name == "dissatisfaction"
    assert mock_detector.text_column == "det_normalized_last_body"


@pytest.mark.usefixtures("mock_detector")
@pytest.mark.parametrize(
    "row, good_deterministic_result",
    [
        (
            {"det_normalized_last_body": "je suis content de votre service."},
            False,
        ),
        (
            {"det_normalized_last_body": "je suis complètement insatisfait de votre service."},
            True,
        ),
        (
            {
                "det_normalized_last_body": "Franchement, j'en ai marre de ce genre de service qui ne respecte pas ses engagements."
            },
            True,
        ),
        (
            {"det_normalized_last_body": "Je suis trop déçu par la qualité, je m'attendais à bien mieux pour ce prix."},
            True,
        ),
        (
            {"det_normalized_last_body": "C'est vraiment décevant de voir un tel manque de professionnalisme."},
            True,
        ),
    ],
)
def test_by_regex_detect(row, good_deterministic_result, mock_detector):
    """Unit test of the transform() method."""
    df_copy = row.copy()
    df_copy = mock_detector.pre_detect(df_copy, debug_mode=True)
    df_copy = mock_detector.by_regex_detect(df_copy, debug_mode=True)

    deterministic_result = mock_detector.DISSATISFACTION_BY_REGEX_MATCH_COL
    deterministic_debug_result = mock_detector.debug_dict_col

    assert deterministic_result in df_copy.keys()
    assert deterministic_debug_result in df_copy.keys()
    assert df_copy[deterministic_result] == good_deterministic_result


@pytest.mark.usefixtures("mock_detector")
@pytest.mark.parametrize(
    "row, good_ml_result",
    [
        (
            {"det_normalized_last_body": "je suis complètement insatisfait de votre service."},
            True,
        ),
        (
            {
                "det_normalized_last_body": "Un service médiocre, avec des frais cachés qui ont presque doublé le coût final. Je ne ferai plus appel à eux."
            },
            True,
        ),
        (
            {
                "det_normalized_last_body": "Très déçu. L’article ne correspond pas du tout à la description, et la qualité laisse à désirer."
            },
            True,
        ),
    ],
)
def test_by_ml_detection(row, good_ml_result, mock_detector):
    """Unit test of the transform() method."""
    df_copy = row.copy()
    # Test result
    df_copy = mock_detector.pre_detect(df_copy, debug_mode=True)
    df_copy = mock_detector.by_ml_detect(df_copy, debug_mode=True)

    # Test result
    ml_result_col = mock_detector.DISSATISFACTION_ML_MATCH_COL
    ml_score_col = mock_detector.DISSATISFACTION_ML_SCORE_COL

    assert ml_result_col in df_copy.keys()
    assert ml_score_col in df_copy.keys()
    assert df_copy[ml_result_col] == good_ml_result
    assert isinstance(df_copy[ml_score_col], float)
    assert df_copy[ml_score_col] > 0.5


@pytest.mark.usefixtures("mock_detector")
@pytest.mark.parametrize(
    "df, good_result",
    [
        (
            DataFrame(
                {
                    "det_normalized_last_body": ["je suis complètement insatisfait de votre service."],
                }
            ),
            True,
        ),
        (
            DataFrame(
                {
                    "det_normalized_last_body": [
                        "Ce retard est une véritable catastrophe, cela m'a causé beaucoup de problèmes."
                    ],
                }
            ),
            True,
        ),
        (
            DataFrame(
                {
                    "det_normalized_last_body": [
                        "Le traitement que j'ai reçu est honteux, surtout venant d'une entreprise comme la vôtre."
                    ],
                }
            ),
            True,
        ),
    ],
)
def test_by_transform_detection(df, good_result, mock_detector):
    """Unit test of the transform() method."""
    df_copy = df.copy()
    # Test result
    df_copy = mock_detector.transform(df_copy)
    # Test result
    result_col = mock_detector.result_column
    assert result_col in df_copy.keys()
    assert bool(df_copy[result_col][0]) == good_result
