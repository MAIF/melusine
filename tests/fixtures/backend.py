import pytest


@pytest.fixture
def backend_base_data():
    return {
        "str_col": ["foo", "bar"],
        "int_col": [1, 2],
    }


def single_input_single_output(value, test_keyword_arg=False):
    return_value = value.upper()
    if test_keyword_arg:
        return_value += "_kwarg"
    return return_value


def single_input_multi_output(value, test_keyword_arg=False):
    return_value = value.upper()
    if test_keyword_arg:
        return_value += "_kwarg"
    return return_value, value.capitalize()


def multi_input_single_output(value1, value2, test_keyword_arg=False):
    return_value = value1
    if test_keyword_arg:
        return_value += "_kwarg"
    return value2 * return_value


def multi_input_multi_output(value1, value2, test_keyword_arg=False):
    return_value = value1.upper()
    if test_keyword_arg:
        return_value += "_kwarg"
    return return_value, value2 * 2


def row_input_single_output(row, test_keyword_arg=False):
    return_value = row["str_col"]
    if test_keyword_arg:
        return_value += "_kwarg"
    return f'{return_value}_{row["int_col"]}'


def row_input_multi_output(row, test_keyword_arg=False):
    return_value = row["str_col"].upper()
    if test_keyword_arg:
        return_value += "_kwarg"
    return return_value, row["int_col"] * 2


def row_input_row_output(row, test_keyword_arg=False):
    return_value = row["str_col"].upper()
    if test_keyword_arg:
        return_value += "_kwarg"
    row["new_str_col"] = return_value
    return row


testcase_single_input_single_output = dict(
    test_name="testcase_single_input_single_output",
    func=single_input_single_output,
    input_columns=["str_col"],
    output_columns=["new_str_col"],
    expected_data={
        "str_col": ["foo", "bar"],
        "int_col": [1, 2],
        "new_str_col": ["FOO", "BAR"],
    },
)

testcase_single_input_single_output_kwarg = dict(
    test_name="testcase_single_input_single_output_kwarg",
    func=single_input_single_output,
    input_columns=["str_col"],
    output_columns=["new_str_col"],
    expected_data={
        "str_col": ["foo", "bar"],
        "int_col": [1, 2],
        "new_str_col": ["FOO_kwarg", "BAR_kwarg"],
    },
    kwargs=dict(test_keyword_arg=True),
)


testcase_single_input_multi_output = dict(
    test_name="testcase_single_input_multi_output",
    func=single_input_multi_output,
    input_columns=["str_col"],
    output_columns=["new_str_col1", "new_str_col2"],
    expected_data={
        "str_col": ["foo", "bar"],
        "int_col": [1, 2],
        "new_str_col1": ["FOO", "BAR"],
        "new_str_col2": ["Foo", "Bar"],
    },
)

testcase_single_input_multi_output_kwarg = dict(
    test_name="testcase_single_input_multi_output_kwarg",
    func=single_input_multi_output,
    input_columns=["str_col"],
    output_columns=["new_str_col1", "new_str_col2"],
    expected_data={
        "str_col": ["foo", "bar"],
        "int_col": [1, 2],
        "new_str_col1": ["FOO_kwarg", "BAR_kwarg"],
        "new_str_col2": ["Foo", "Bar"],
    },
    kwargs=dict(test_keyword_arg=True),
)

testcase_row_input_single_output = dict(
    test_name="testcase_row_input_single_output",
    func=row_input_single_output,
    input_columns=["str_col", "int_col"],
    output_columns=["new_str_col"],
    expected_data={
        "str_col": ["foo", "bar"],
        "int_col": [1, 2],
        "new_str_col": ["foo_1", "bar_2"],
    },
)

testcase_row_input_single_output_kwarg = dict(
    test_name="testcase_row_input_single_output_kwarg",
    func=row_input_single_output,
    input_columns=["str_col", "int_col"],
    output_columns=["new_str_col"],
    expected_data={
        "str_col": ["foo", "bar"],
        "int_col": [1, 2],
        "new_str_col": ["foo_kwarg_1", "bar_kwarg_2"],
    },
    kwargs=dict(test_keyword_arg=True),
)

testcase_no_input_single_output = dict(
    test_name="testcase_no_input_single_output",
    func=row_input_single_output,
    input_columns=None,
    output_columns=["new_str_col"],
    expected_data={
        "str_col": ["foo", "bar"],
        "int_col": [1, 2],
        "new_str_col": ["foo_1", "bar_2"],
    },
)

testcase_no_input_single_output_kwarg = dict(
    test_name="testcase_no_input_single_output_kwarg",
    func=row_input_single_output,
    input_columns=None,
    output_columns=["new_str_col"],
    expected_data={
        "str_col": ["foo", "bar"],
        "int_col": [1, 2],
        "new_str_col": ["foo_kwarg_1", "bar_kwarg_2"],
    },
    kwargs=dict(test_keyword_arg=True),
)

testcase_multi_input_no_output = dict(
    test_name="testcase_multi_input_no_output",
    func=row_input_row_output,
    input_columns=["str_col", "int_col"],
    output_columns=None,
    expected_data={
        "str_col": ["foo", "bar"],
        "int_col": [1, 2],
        "new_str_col": ["FOO", "BAR"],
    },
)

testcase_multi_input_no_output_kwarg = dict(
    test_name="testcase_multi_input_no_output_kwarg",
    func=row_input_row_output,
    input_columns=["str_col", "int_col"],
    output_columns=None,
    expected_data={
        "str_col": ["foo", "bar"],
        "int_col": [1, 2],
        "new_str_col": ["FOO_kwarg", "BAR_kwarg"],
    },
    kwargs=dict(test_keyword_arg=True),
)

testcase_row_input_row_output = dict(
    test_name="testcase_row_input_row_output",
    func=row_input_row_output,
    input_columns=None,
    output_columns=None,
    expected_data={
        "str_col": ["foo", "bar"],
        "int_col": [1, 2],
        "new_str_col": ["FOO", "BAR"],
    },
)

testcase_row_input_row_output_kwarg = dict(
    test_name="testcase_row_input_row_output_kwarg",
    func=row_input_row_output,
    input_columns=None,
    output_columns=None,
    expected_data={
        "str_col": ["foo", "bar"],
        "int_col": [1, 2],
        "new_str_col": ["FOO_kwarg", "BAR_kwarg"],
    },
    kwargs=dict(test_keyword_arg=True),
)

testcase_no_input_multi_output = dict(
    test_name="testcase_no_input_multi_output",
    func=row_input_multi_output,
    input_columns=None,
    output_columns=["new_str_col", "new_int_col"],
    expected_data={
        "str_col": ["foo", "bar"],
        "int_col": [1, 2],
        "new_str_col": ["FOO", "BAR"],
        "new_int_col": [2, 4],
    },
)

testcase_no_input_multi_output_kwarg = dict(
    test_name="testcase_no_input_multi_output_kwarg",
    func=row_input_multi_output,
    input_columns=None,
    output_columns=["new_str_col", "new_int_col"],
    expected_data={
        "str_col": ["foo", "bar"],
        "int_col": [1, 2],
        "new_str_col": ["FOO_kwarg", "BAR_kwarg"],
        "new_int_col": [2, 4],
    },
    kwargs=dict(test_keyword_arg=True),
)

testcase_row_input_multi_output = dict(
    test_name="testcase_row_input_multi_output",
    func=row_input_multi_output,
    input_columns=["str_col", "int_col"],
    output_columns=["new_str_col", "new_int_col"],
    expected_data={
        "str_col": ["foo", "bar"],
        "int_col": [1, 2],
        "new_str_col": ["FOO", "BAR"],
        "new_int_col": [2, 4],
    },
)

testcase_row_input_multi_output_kwarg = dict(
    test_name="testcase_row_input_multi_output_kwarg",
    func=row_input_multi_output,
    input_columns=["str_col", "int_col"],
    output_columns=["new_str_col", "new_int_col"],
    expected_data={
        "str_col": ["foo", "bar"],
        "int_col": [1, 2],
        "new_str_col": ["FOO_kwarg", "BAR_kwarg"],
        "new_int_col": [2, 4],
    },
    kwargs=dict(test_keyword_arg=True),
)


testcase_list = [value for key, value in locals().items() if key.startswith("testcase")]


def get_fixture_name(fixture_value):
    return fixture_value.get("test_name", "missing_test_name")


@pytest.fixture(
    params=testcase_list,
    ids=get_fixture_name,
)
def backend_testcase(request):
    """Fixture to test all backend configurations"""
    testcase = request.param
    return testcase
