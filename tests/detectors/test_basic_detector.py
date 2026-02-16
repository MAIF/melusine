import pandas as pd
import pytest

from melusine.base import MelusineDetector

TEST_RESULT = "test_result"


class DummyDetector(MelusineDetector):
    def __init__(self):
        """Dummy"""
        super().__init__(name="dummy", input_columns=["text"], output_columns=["result"])

    def pre_detect(self, row, debug_mode=False):
        """Dummy"""
        # Simule un comportement simple pour le test
        if debug_mode:
            row[self.debug_dict_col]["test_debug"] = True

        row["result"] = TEST_RESULT
        return row

    def detect(self, row, debug_mode=False):
        """Dummy"""
        return row

    def post_detect(self, row, debug_mode=False):
        """Dummy"""
        return row


@pytest.mark.usefixtures("use_dict_backend")
def test_transform_with_dict_and_debug_mode():
    detector = DummyDetector()
    data_dict = {"text": "exemple"}
    result = detector.transform(data_dict, debug_mode=True)
    assert isinstance(result, dict)
    assert result[detector.debug_dict_col]["test_debug"] == True
    assert result["result"] == TEST_RESULT


@pytest.mark.usefixtures("use_dict_backend")
def test_transform_with_dict_and_legacy_debug_mode():
    detector = DummyDetector()
    data_dict = {"text": "exemple", "debug": True}
    result = detector.transform(data_dict)
    assert isinstance(result, dict)
    assert result[detector.debug_dict_col]["test_debug"] == True
    assert result["result"] == TEST_RESULT


def test_transform_with_dataframe_and_debug_mode():
    detector = DummyDetector()
    data_df = pd.DataFrame([{"text": "exemple"}])
    result = detector.transform(data_df, debug_mode=True)
    assert isinstance(result, pd.DataFrame)
    assert result[detector.debug_dict_col].iloc[0]["test_debug"] == True
    assert result["result"].iloc[0] == TEST_RESULT


def test_transform_with_dataframe_and_legacy_debug_mode():
    detector = DummyDetector()
    data_df = pd.DataFrame([{"text": "exemple"}])
    data_df.debug = True
    result = detector.transform(data_df)
    assert isinstance(result, pd.DataFrame)
    assert result[detector.debug_dict_col].iloc[0]["test_debug"] == True
    assert result["result"].iloc[0] == TEST_RESULT
