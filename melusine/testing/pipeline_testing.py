"""
Module that contains utility functions for tests (in /tests).
"""
from typing import Any, Dict

from melusine.base import MelusineTransformer
from melusine.pipeline import MelusinePipeline

expected_suffix: str = "_expected"


def assert_pipeline_results(email: Dict[str, Any], pipeline_name: str) -> None:
    """
    Assert that the pipeline execution result correspond to the testcase expectation.

    Parameters
    ----------
    email: Dict[str, Any]
        Email content.
    pipeline_name: str
        Name of a Melusine pipeline.
    """
    # Instantiate Pipeline
    pipeline: MelusinePipeline = MelusinePipeline.from_config(pipeline_name)

    # Useful fields
    test_name: str = email["test_name"]

    # Loop on pipeline transformers
    for transformer_name, transformer in pipeline.steps:
        email = assert_transformation(email, transformer, transformer_name, test_name)

    # Check that the pipeline returns a dict
    assert isinstance(email, dict)

    # Look for untested fields
    untested_fields = [x for x in email if x.endswith(expected_suffix)]
    assert not untested_fields, f"Field(s) {untested_fields} have not been tested for" f"Test-case '{test_name}'"


def assert_transformation(
    email: Dict[str, Any], transformer: MelusineTransformer, transformer_name: str, test_name: str
) -> Dict[str, Any]:
    """

    Parameters
    ----------
    email: Dict[str, Any]
        Email data dict
    test_name: str
        Name of the current test
    transformer: MelusineTransformer
        Data transformer instance
    transformer_name: str
        Name of the current transformer

    Returns
    -------
    email: Dict[str, Any]
        Transformed email data dict
    """
    # Apply transformer on email
    email = transformer.transform(email)

    # Do we have an expected value for this transformer?
    expected_key = f"{transformer_name}{expected_suffix}"
    if expected_key in email:
        expectation_dict: Dict[str, Any] = email.pop(expected_key)

        # Loop on columns with an expected value
        for col, expected_value in expectation_dict.items():
            # Specific case for the message column
            if col.startswith("messages"):
                assert_message_attribute(col, email, expected_value, test_name, transformer_name)

            # Regular case
            else:
                assert email[col] == expected_value, (
                    f"Failure for test {test_name} at step {transformer_name}. "
                    f"Value expected for column {col} : {expected_value}. "
                    f"Value obtained for column {col} : {email[col]}"
                )
    return email


def assert_message_attribute(
    col: str, email: Dict[str, Any], expected_value: Any, test_name: str, transformer_name: str
) -> None:
    """

    Parameters
    ----------
    col: str
        Column name
    email: Dict[str, Any]
        Email data dict
    expected_value: Any
        Expected message attribute value
    test_name: str
        Name of the current test
    transformer_name: str
        Name of the current transformer

    Returns
    -------

    """
    # Check the number of messages in the conversation
    assert len(email["messages"]) == len(expected_value)

    # Loop on messages
    for message, expected_attr in zip(email["messages"], expected_value):
        _, attribute_name = col.rsplit(".")
        attribute_value = getattr(message, attribute_name)

        # Test attribute values
        assert attribute_value == expected_attr, (
            f"Failure for test {test_name} at step {transformer_name}. "
            f"Value expected for attribute {attribute_name} : {expected_attr}. "
            f"Value obtained for attribute {attribute_name} : {attribute_value}"
        )
