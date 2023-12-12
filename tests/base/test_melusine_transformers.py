import pandas as pd
import pytest

from melusine.base import MelusineTransformer, TransformError


class FakeClass(MelusineTransformer):
    def __init__(self):
        super().__init__(input_columns="input_col", output_columns="output_col", func=self.my_method)

    def my_method(self, df, debug_mode=False):
        raise ValueError


def test_transform_error():
    df = pd.DataFrame([{"input_col": "test"}])
    instance = FakeClass()

    with pytest.raises(TransformError, match="FakeClass.*my_method.*input_col"):
        _ = instance.transform(df)


def test_from_config_from_key_error():
    """Unit test"""

    with pytest.raises(ValueError):
        _ = MelusineTransformer.from_config(config_key="key", config_dict={"foo": "bar"})


def test_missing_func():
    """Unit test"""
    df = pd.DataFrame([{"a": [1, 2, 3]}])
    transformer = MelusineTransformer(input_columns=["a"], output_columns=["b"])

    with pytest.raises(AttributeError):
        transformer.transform(df)
