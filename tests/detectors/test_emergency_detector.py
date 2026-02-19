"""
Unit test of the EmergencyDetector.

"""

import pandas as pd
import pytest

from melusine.detectors import EmergencyDetector


@pytest.fixture
def emergency_detector_df():
    m0_debug_expected = {
        "match_result": True,
        "negative_match_data": {},
        "neutral_match_data": {},
        "positive_match_data": {"DEFAULT": [{"match_text": "urgent", "start": 10, "stop": 16}]},
    }

    df = pd.DataFrame(
        {
            "header": ["hey"],
            "det_normalized_last_body": ["c'est urgent"],
            "debug_expectation": [m0_debug_expected],
        }
    )

    return df


def test_emergency_detector(emergency_detector_df):
    """Unit test of the debug mode."""
    df = emergency_detector_df
    detector = EmergencyDetector(
        name="emergency",
    )
    result_col = detector.result_column
    debug_dict_col = detector.debug_dict_col

    # Transform data
    df = detector.transform(df, debug_mode=True)

    # Test result
    assert result_col in df.columns
    assert debug_dict_col in df.columns

    for i, row in df.iterrows():
        assert row[debug_dict_col][detector.regex.regex_name] == row["debug_expectation"]
