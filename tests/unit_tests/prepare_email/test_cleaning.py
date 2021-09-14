import pytest
import numpy as np
from melusine.prepare_email.cleaning import (
    remove_multiple_spaces_and_strip_text,
    remove_accents,
    remove_transfer_answer_header,
)


@pytest.mark.parametrize(
    "input_str, expected_str",
    [
        ("hello   world", "hello world"),
        ("\n   hello world    ", "hello world"),
        ("----- hello\tworld *****", "hello world"),
        ("hello-world", "hello-world"),
        ("hello - world", "hello world"),
    ],
)
def test_remove_multiple_spaces_and_strip_text(input_str, expected_str):
    result = remove_multiple_spaces_and_strip_text(input_str)
    np.testing.assert_string_equal(result, expected_str)


def test_remove_accents():
    input_str = "éèëêàù"
    expected_str = "eeeeau"

    result = remove_accents(input_str)
    np.testing.assert_string_equal(result, expected_str)


@pytest.mark.parametrize(
    "input_str, expected_str",
    [
        ("RE: hello world", " hello world"),
        ("re :", ""),
        ("TR: hello", " hello"),
        ("hello ", "hello "),
        ("Fwd:hello", "hello"),
    ],
)
def test_remove_transfer_answer_header(input_str, expected_str):
    result = remove_transfer_answer_header(input_str)
    np.testing.assert_string_equal(result, expected_str)
