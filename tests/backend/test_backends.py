import numpy as np
import pandas as pd
import pytest

from melusine.backend import backend
from melusine.backend.dict_backend import DictBackend
from melusine.backend.pandas_backend import PandasBackend
from melusine.processors import Normalizer


def test_reset_backend():
    """Test"""
    dict_data = {"input_col": "àçöbïù"}
    df_data = pd.DataFrame([dict_data])
    processor = Normalizer(input_columns="input_col", output_columns="output_col")

    dict_backend = DictBackend()
    backend.reset(dict_backend)
    dict_out = processor.transform(dict_data)
    assert isinstance(dict_out, dict)

    backend.reset()
    df_out = processor.transform(df_data)
    assert isinstance(df_out, pd.DataFrame)


def test_unknown_backend():
    with pytest.raises(ValueError):
        backend.reset("unknown")


def test_backend_error():
    backend._backend = None
    with pytest.raises(AttributeError):
        _ = backend.backend
    backend.reset()


def test_check_debug_flag_dict():
    dict_data = {"input_col": "àçöbïù"}
    dict_backend = DictBackend()

    assert dict_backend.check_debug_flag(dict_data) is False

    dict_data = {"debug": False}
    assert dict_backend.check_debug_flag(dict_data) is False

    dict_data = {"debug": True}
    assert dict_backend.check_debug_flag(dict_data) is True


def test_check_debug_flag_pandas():
    df = pd.DataFrame([{"input_col": "àçöbïù"}])
    pandas_backend = PandasBackend()

    assert pandas_backend.check_debug_flag(df) is False

    df.debug = False
    assert pandas_backend.check_debug_flag(df) is False

    df.debug = True
    assert pandas_backend.check_debug_flag(df) is True


def test_add_fields_dict():
    dict_data1 = {"col1": 1, "col8": 8}
    dict_data2 = {"col1": 10, "col2": 2}
    dict_backend = DictBackend()

    data = dict_backend.add_fields(dict_data1, dict_data2)
    assert data == {"col1": 10, "col2": 2, "col8": 8}


def test_dict_backend(backend_base_data, backend_testcase):
    """Test"""
    expected_data = backend_testcase["expected_data"]
    func = backend_testcase["func"]
    input_columns = backend_testcase["input_columns"]
    output_columns = backend_testcase["output_columns"]
    kwargs = backend_testcase.get("kwargs", dict())

    dict_backend = DictBackend()
    n_values = len(list(backend_base_data.values())[0])

    for i in range(n_values):
        data_dict = {key: backend_base_data[key][i] for key in backend_base_data}
        expected_data_dict = {key: expected_data[key][i] for key in expected_data}

        data_dict_transform = dict_backend.apply_transform(
            data=data_dict, func=func, input_columns=input_columns, output_columns=output_columns, **kwargs
        )
        assert data_dict_transform == expected_data_dict


def test_dict_backend_impossible_situation():
    dict_backend = DictBackend()
    with pytest.raises(ValueError):
        _ = dict_backend.apply_transform(
            data={"a": 0}, func=lambda x: x + 1, input_columns=["int_col"], output_columns=None
        )


@pytest.mark.parametrize("progress_bar", [False, True])
def test_pandas_backend(backend_base_data, backend_testcase, progress_bar):
    """Test"""
    expected_data = backend_testcase["expected_data"]
    func = backend_testcase["func"]
    input_columns = backend_testcase["input_columns"]
    output_columns = backend_testcase["output_columns"]
    kwargs = backend_testcase.get("kwargs", dict())

    pandas_backend = PandasBackend(progress_bar=progress_bar, workers=1)

    df_base = pd.DataFrame(backend_base_data)

    df_expected = pd.DataFrame(expected_data)
    df_transform = pandas_backend.apply_transform(
        data=df_base, func=func, input_columns=input_columns, output_columns=output_columns, **kwargs
    )
    pd.testing.assert_frame_equal(df_transform, df_expected)


def test_pandas_backend_multiprocess(backend_base_data, backend_testcase):
    """Test"""
    expected_data = backend_testcase["expected_data"]
    func = backend_testcase["func"]
    input_columns = backend_testcase["input_columns"]
    output_columns = backend_testcase["output_columns"]
    kwargs = backend_testcase.get("kwargs", dict())

    pandas_backend = PandasBackend(progress_bar=False, workers=2)

    # Test on a small dataset (does not trigger multiprocessing)
    df_base = pd.DataFrame(backend_base_data)
    df_expected = pd.DataFrame(expected_data)

    df_transform = pandas_backend.apply_transform(
        data=df_base, func=func, input_columns=input_columns, output_columns=output_columns, **kwargs
    )
    pd.testing.assert_frame_equal(df_transform, df_expected)

    # Augment dataset size to trigger multiprocessing (preserve column type)
    df_long = pd.DataFrame({col: value for col in df_base for value in [np.repeat(df_base[col].values, 3, axis=0)]})
    df_expected_long = pd.DataFrame(
        {col: value for col in df_expected for value in [np.repeat(df_expected[col].values, 3, axis=0)]}
    )

    df_transform = pandas_backend.apply_transform(
        data=df_long, func=func, input_columns=input_columns, output_columns=output_columns, **kwargs
    )
    pd.testing.assert_frame_equal(df_transform, df_expected_long)
