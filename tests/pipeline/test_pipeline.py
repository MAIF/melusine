"""
Unit test for pipeline.py
"""
import pandas as pd
import pytest

from melusine.base import MelusineTransformer
from melusine.pipeline import MelusinePipeline

# Dummy variables
dum0 = 42
dum1 = "dum"
dum2 = "dumdum"
dum3 = "duuum"


# Define dummy processors
class DummyProcessor(MelusineTransformer):
    def __init__(self, input_columns="a", output_columns=("b",), dummy_attr=dum1):
        super().__init__(input_columns, output_columns, func=self.add_dummy_col)
        self.dummy_attr = dummy_attr

    def add_dummy_col(self, col_a_data):
        return self.dummy_attr


def test_pipeline_with_processors():
    d1 = DummyProcessor()
    d2 = DummyProcessor(output_columns=("c",), dummy_attr=dum2)

    # Create pipeline
    pipe = MelusinePipeline(steps=[("d1", d1), ("d2", d2)], verbose=True)

    # Create data
    df = pd.DataFrame({"a": [dum0, dum0]})

    # Fit the pipeline and transform the data
    df_transformed = pipe.transform(df)

    # Most basic test, check that the pipeline returns a pandas DataFrame
    assert isinstance(df_transformed, pd.DataFrame)
    assert "a" in df_transformed.columns
    assert "b" in df_transformed.columns
    assert "c" in df_transformed.columns

    assert df_transformed["a"].iloc[0] == dum0
    assert df_transformed["b"].iloc[0] == dum1
    assert df_transformed["c"].iloc[0] == dum2


def test_meta_pipeline():
    d1 = DummyProcessor()
    d2 = DummyProcessor(output_columns=("c",), dummy_attr=dum2)
    d3 = DummyProcessor(output_columns=("d",), dummy_attr=dum3)

    # Create pipeline
    pipe1 = MelusinePipeline(steps=[("d1", d1), ("d2", d2)], verbose=True)
    pipe2 = MelusinePipeline(steps=[("d3", d3)])
    meta_pipe = MelusinePipeline(steps=[("pipe1", pipe1), ("pipe2", pipe2)])

    # Create data
    df = pd.DataFrame({"a": [dum0, dum0]})

    # Fit the pipeline and transform the data
    df_transformed = meta_pipe.transform(df)

    # Most basic test, check that the pipeline returns a pandas DataFrame
    assert isinstance(df_transformed, pd.DataFrame)
    assert "a" in df_transformed.columns
    assert "b" in df_transformed.columns
    assert "c" in df_transformed.columns
    assert "d" in df_transformed.columns

    assert df_transformed["a"].iloc[0] == dum0
    assert df_transformed["b"].iloc[0] == dum1
    assert df_transformed["c"].iloc[0] == dum2
    assert df_transformed["d"].iloc[0] == dum3


def test_pipeline_from_config():
    _ = MelusinePipeline.from_config(config_key="my_pipeline")


def test_pipeline_from_config_error():
    with pytest.raises(ValueError, match=r"'config_key' and 'config_dict'"):
        _ = MelusinePipeline.from_config(config_key="x", config_dict={"a": 5})


def test_pipeline_get_config_from_key():
    conf = MelusinePipeline.get_config_from_key(config_key="my_pipeline")
    assert conf


def test_pipeline_from_config_missing_class():
    config_dict = {
        "steps": [
            {
                "class_name": "UnknownClass",
                "module": "melusine.processors",
                "name": "test",
                "parameters": {
                    "mel": "usine",
                },
            },
        ]
    }
    with pytest.raises(AttributeError, match=r"UnknownClass.*melusine.processors"):
        _ = MelusinePipeline.from_config(config_dict=config_dict)


def test_pipeline_input_output_columns():
    d1 = DummyProcessor(input_columns=("a",), output_columns=("b",))
    d2 = DummyProcessor(input_columns=("b",), output_columns=("c",), dummy_attr=dum2)

    # Create pipeline
    pipe = MelusinePipeline(steps=[("d1", d1), ("d2", d2)], verbose=True)

    assert len(pipe.input_columns) == 2
    assert "a" in pipe.input_columns
    assert "b" in pipe.input_columns

    assert len(pipe.output_columns) == 2
    assert "b" in pipe.output_columns
    assert "c" in pipe.output_columns
