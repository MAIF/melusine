import pytest

from melusine.testing import assert_pipeline_results

from .test_emails_fixtures import testcase


# The test_message fixture sequentially takes the value of the
# test_cases defined in melusine_code/tests/fixtures/test_emails_fixtures.py
@pytest.mark.usefixtures("use_dict_backend")
def test_pipeline_steps(testcase):
    # Run pipeline tests
    pipeline_name = testcase.pop("pipeline")
    assert_pipeline_results(testcase, pipeline_name)
