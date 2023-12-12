"""
Unit test of the ThanksDetector.

"""
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

from melusine.base import MissingFieldError
from melusine.detectors import ThanksDetector
from melusine.message import Message


@pytest.fixture
def thanks_detector_df():
    m0 = Message("")
    m0.tags = [
        ("HELLO", "Bonjour"),
        ("THANKS", "Merci beaucoup"),
    ]
    m0_messages = [m0]
    m0_expected = True
    m0_debug_expected = {
        "match_result": True,
        "negative_match_data": {},
        "neutral_match_data": {},
        "positive_match_data": {"DEFAULT": [{"match_text": "Merci", "start": 0, "stop": 5}]},
    }

    m1 = Message("")
    m1.tags = [
        ("HELLO", "Bonjour"),
        ("THANKS", "Merci, j'attends une reponse"),
    ]
    m1_messages = [m1]
    m1_expected = False
    m1_debug_expected = {
        "match_result": False,
        "negative_match_data": {"FORBIDDEN_WORDS": [{"match_text": "attend", "start": 9, "stop": 15}]},
        "neutral_match_data": {},
        "positive_match_data": {"DEFAULT": [{"match_text": "Merci", "start": 0, "stop": 5}]},
    }

    df = pd.DataFrame(
        {
            "messages": [m0_messages, m1_messages],
            "detection_expectation": [m0_expected, m1_expected],
            "debug_expectation": [m0_debug_expected, m1_debug_expected],
        }
    )

    return df


def test_thanks_detector(thanks_detector_df):
    """Unit test of the debug mode."""
    df = thanks_detector_df
    df_copy = df.copy()
    detector = ThanksDetector(
        name="thanks",
    )
    result_col = detector.result_column
    debug_dict_col = detector.debug_dict_col

    # Transform data
    df.debug = True
    df = detector.transform(df)

    # Test result
    assert result_col in df.columns
    assert debug_dict_col in df.columns

    for i, row in df.iterrows():
        assert row[result_col] == row["detection_expectation"]
        assert row[debug_dict_col][detector.thanks_regex.regex_name] == row["debug_expectation"]


def test_thanks_detector_missing_field(thanks_detector_df):
    """Unit test of the debug mode."""
    df = thanks_detector_df.copy()

    detector = ThanksDetector(
        name="thanks",
    )
    df = df.drop(detector.input_columns, axis=1)

    # Transform data
    with pytest.raises(MissingFieldError, match=str(detector.input_columns)):
        _ = detector.transform(df)


@pytest.mark.parametrize(
    "tags, has_body, thanks_text, thanks_parts",
    [
        (
            [
                ("HELLO", "Bonjour madame"),
                ("BODY", "Voici le dossier"),
                ("THANKS", "Merci a vous"),
            ],
            True,
            "Merci a vous",
            [("THANKS", "Merci a vous")],
        ),
        (
            [
                ("HELLO", "Bonjour madame"),
                ("THANKS", "Merci"),
                ("THANKS", "Merci a vous"),
            ],
            False,
            "Merci\nMerci a vous",
            [("THANKS", "Merci"), ("THANKS", "Merci a vous")],
        ),
    ],
)
@pytest.mark.usefixtures("use_dict_backend")
def test_thanks_detector_debug(tags, has_body, thanks_text, thanks_parts):
    """Unit test of the debug mode."""

    data = {
        "messages": [Message(text="", tags=tags)],
        "debug": True,
    }

    detector = ThanksDetector(
        name="thanks",
    )

    # Transform data
    data = detector.transform(data)

    # Test result
    assert "debug_thanks" in data
    assert "has_body" in data["debug_thanks"]
    assert "thanks_text" in data["debug_thanks"]
    assert "thanks_parts" in data["debug_thanks"]

    assert data["debug_thanks"]["has_body"] == has_body
    assert data["debug_thanks"]["thanks_text"] == thanks_text
    assert data["debug_thanks"]["thanks_parts"] == thanks_parts
