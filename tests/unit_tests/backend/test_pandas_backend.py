import pytest
import pandas as pd
from melusine.backend.active_backend import backend, switch_backend
from pandas.testing import assert_series_equal


@pytest.fixture
def dict_single_input_single_output():
    return {
        "input": 2,
        "expected_output": 6,
        "expected_output_kwargs": 10,
    }


@pytest.fixture
def dataframe_single_input_single_output(dict_single_input_single_output):
    return pd.DataFrame(
        [dict_single_input_single_output, dict_single_input_single_output]
    )


@pytest.mark.parametrize("progress_bar", [True, False])
def test_pandas_backend_single_input_single_output(
    progress_bar,
    dataframe_single_input_single_output,
):
    df = dataframe_single_input_single_output.copy()
    switch_backend("pandas", progress_bar=progress_bar)

    def f_single_input_single_output(x, n=3):
        return x * n

    input_columns = ("input",)
    output_columns = ("output",)

    # Test default arguments
    df = backend.apply_transform(
        df,
        f_single_input_single_output,
        input_columns=input_columns,
        output_columns=output_columns,
    )

    assert_series_equal(df["output"], df["expected_output"], check_names=False)

    # Test kwargs
    df = backend.apply_transform(
        df,
        f_single_input_single_output,
        input_columns=input_columns,
        output_columns=output_columns,
        n=5,
    )

    assert_series_equal(df["output"], df["expected_output_kwargs"], check_names=False)


@pytest.fixture
def dict_single_input_multi_output():
    return {
        "input": 2,
        "expected_output1": 6,
        "expected_output2": 20,
        "expected_output_kwargs1": 10,
        "expected_output_kwargs2": 20,
    }


@pytest.fixture
def dataframe_single_input_multi_output(dict_single_input_multi_output):
    return pd.DataFrame(
        [dict_single_input_multi_output, dict_single_input_multi_output]
    )


@pytest.mark.parametrize("progress_bar", [True, False])
def test_pandas_backend_single_input_multi_output(
    progress_bar, dataframe_single_input_multi_output
):
    df = dataframe_single_input_multi_output.copy()
    switch_backend("pandas", progress_bar=progress_bar)

    def f_single_input_multi_output(x, n=3):
        return x * n, x * 10

    input_columns = ("input",)
    output_columns = ("output1", "output2")

    # Test default arguments
    df = backend.apply_transform(
        df,
        f_single_input_multi_output,
        input_columns=input_columns,
        output_columns=output_columns,
    )

    assert_series_equal(df["output1"], df["expected_output1"], check_names=False)
    assert_series_equal(df["output2"], df["expected_output2"], check_names=False)

    # Test kwargs
    df = backend.apply_transform(
        df,
        f_single_input_multi_output,
        input_columns=input_columns,
        output_columns=output_columns,
        n=5,
    )

    assert_series_equal(df["output1"], df["expected_output_kwargs1"], check_names=False)
    assert_series_equal(df["output2"], df["expected_output_kwargs2"], check_names=False)


@pytest.fixture
def dict_multi_input_single_output():
    return {
        "input1": 2,
        "input2": 3,
        "expected_output": 8,
        "expected_output_kwargs": 10,
    }


@pytest.fixture
def dataframe_multi_input_single_output(dict_multi_input_single_output):
    return pd.DataFrame(
        [dict_multi_input_single_output, dict_multi_input_single_output]
    )


@pytest.mark.parametrize("progress_bar", [True, False])
def test_pandas_backend_multi_input_single_output(
    progress_bar,
    dataframe_multi_input_single_output,
):
    df = dataframe_multi_input_single_output.copy()
    switch_backend("pandas", progress_bar=progress_bar)

    def f_multi_input_single_output(row, n=3):
        return row["input1"] + row["input2"] + n

    input_columns = ("input1", "input2")
    output_columns = ("output",)

    # Test default arguments
    df = backend.apply_transform(
        df,
        f_multi_input_single_output,
        input_columns=input_columns,
        output_columns=output_columns,
    )

    assert_series_equal(df["output"], df["expected_output"], check_names=False)

    # Test kwargs
    df = backend.apply_transform(
        df,
        f_multi_input_single_output,
        input_columns=input_columns,
        output_columns=output_columns,
        n=5,
    )

    assert_series_equal(df["output"], df["expected_output_kwargs"], check_names=False)


@pytest.fixture
def dict_multi_input_multi_output():
    return {
        "input1": 2,
        "input2": 3,
        "expected_output1": 8,
        "expected_output2": 20,
        "expected_output_kwargs1": 10,
        "expected_output_kwargs2": 20,
    }


@pytest.fixture
def dataframe_multi_input_multi_output(dict_multi_input_multi_output):
    return pd.DataFrame([dict_multi_input_multi_output, dict_multi_input_multi_output])


@pytest.mark.parametrize("progress_bar", [True, False])
def test_pandas_backend_multi_input_multi_output(
    progress_bar, dataframe_multi_input_multi_output
):
    df = dataframe_multi_input_multi_output.copy()
    switch_backend("pandas", progress_bar=progress_bar)

    def f_single_multi_input_multi_output(row, n=3):
        return row["input1"] + row["input2"] + n, row["input1"] * 10

    input_columns = ("input1", "input2")
    output_columns = ("output1", "output2")

    # Test default arguments
    df = backend.apply_transform(
        df,
        f_single_multi_input_multi_output,
        input_columns=input_columns,
        output_columns=output_columns,
    )

    assert_series_equal(df["output1"], df["expected_output1"], check_names=False)
    assert_series_equal(df["output2"], df["expected_output2"], check_names=False)

    # Test kwargs
    df = backend.apply_transform(
        df,
        f_single_multi_input_multi_output,
        input_columns=input_columns,
        output_columns=output_columns,
        n=5,
    )

    assert_series_equal(df["output1"], df["expected_output_kwargs1"], check_names=False)
    assert_series_equal(df["output2"], df["expected_output_kwargs2"], check_names=False)
