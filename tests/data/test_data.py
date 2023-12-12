import pytest

from melusine.data import load_email_data


@pytest.mark.parametrize(
    "type, expected_column",
    [("raw", "body"), ("preprocessed", "tokens"), ("full", "tokens")],
)
def test_load_data(type, expected_column):

    df = load_email_data(type=type)
    assert expected_column in df


def test_load_data_error():

    with pytest.raises(ValueError):
        _ = load_email_data(type="unsupported_type")
