import pytest

from melusine.testing.pipeline_testing import assert_pipeline_results


@pytest.mark.usefixtures("use_dict_backend")
def test_pipeline_testing():
    testcase = {
        "test_name": "Simple test",
        "body": "Hello\r\nGood bye",
        "from": "test@gmail.com",
        "header": "Test header",
        "body_cleaner_expected": {"tmp_clean_body": "Hello\nGood bye"},
    }

    assert_pipeline_results(testcase, "my_pipeline")


@pytest.mark.usefixtures("use_dict_backend")
def test_pipeline_testing_error():
    test_name = "Expected result error"
    pipeline_step_name = "body_cleaner"
    field_name = "tmp_clean_body"
    expected_value = "NotTheRightText"

    testcase = {
        "test_name": test_name,
        "body": "Hello\r\nGood bye",
        "from": "test@gmail.com",
        "header": "Test header",
        f"{pipeline_step_name}_expected": {field_name: expected_value},
    }

    with pytest.raises(AssertionError, match=f"{test_name}.*{pipeline_step_name}.*{field_name}.*{expected_value}"):
        assert_pipeline_results(testcase, "my_pipeline")


@pytest.mark.usefixtures("use_dict_backend")
def test_pipeline_testing_untested_field():
    test_name = "Untested field test"
    pipeline_step_name = "non_existent_step"

    testcase = {
        "test_name": test_name,
        "body": "Hello\r\nGood bye",
        "from": "test@gmail.com",
        "header": "Test header",
        f"{pipeline_step_name}_expected": {"field", "expected_value"},
    }

    with pytest.raises(AssertionError, match=f"{pipeline_step_name}.*{test_name}"):
        assert_pipeline_results(testcase, "my_pipeline")
