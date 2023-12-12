"""
Unit tests of the ReplyDetector.
"""
import pandas as pd
import pytest
from pandas import DataFrame

from melusine.detectors import ReplyDetector
from melusine.pipeline import MelusinePipeline


def test_instantiation():
    """Instanciation base test."""

    # Instantiate manually a detector
    detector = ReplyDetector(
        name="reply",
        header_column="clean_header",
    )
    assert isinstance(detector, ReplyDetector)


@pytest.mark.parametrize(
    "row, good_result",
    [
        (
            {"reply_text": "Devis habitation"},
            False,
        ),
        (
            {"reply_text": "tr: Devis habitation"},
            False,
        ),
        (
            {"reply_text": "re: Envoi d'un document de la Société Imaginaire"},
            True,
        ),
        (
            {"reply_text": "re : Virement"},
            True,
        ),
        (
            {"reply_text": ""},
            False,
        ),
    ],
)
def test_deterministic_detect(row, good_result):
    """Method base test."""

    # Instanciate manually a detector
    detector = ReplyDetector(
        name="reply",
        header_column="clean_header",
    )
    # Test method
    row = detector.detect(row)
    res = row[detector.result_column]
    assert res == good_result


@pytest.mark.parametrize(
    "df_emails, expected_result",
    [
        (
            DataFrame(
                {
                    "clean_header": ["Re: Suivi de dossier"],
                }
            ),
            True,
        ),
        (
            DataFrame(
                {
                    "clean_header": ["Suivi de dossier"],
                }
            ),
            False,
        ),
        (
            DataFrame(
                {
                    "clean_header": ["Tr: Suivi de dossier"],
                }
            ),
            False,
        ),
        (
            DataFrame(
                {
                    "clean_header": [""],
                }
            ),
            False,
        ),
    ],
)
def test_transform(df_emails, expected_result):
    """Unit test of the transform() method."""

    # Copy for later load/save test
    df_copy = df_emails.copy()

    # Instantiate manually a detector
    detector = ReplyDetector(
        name="reply",
        header_column="clean_header",
    )

    # Get result column name
    res_col = detector.result_column

    # Apply the detector on data
    df_emails = detector.transform(df_emails)

    # Verify result
    result = df_emails[res_col][0]
    assert result == expected_result


@pytest.mark.parametrize(
    "df_emails, expected_result, expected_debug_info",
    [
        (
            DataFrame(
                {
                    "clean_header": ["Re: Suivi de dossier"],
                }
            ),
            True,
            {
                "ReplyRegex": {
                    "match_result": True,
                    "negative_match_data": {},
                    "neutral_match_data": {},
                    "positive_match_data": {"DEFAULT": [{"match_text": "re:", "start": 0, "stop": 3}]},
                },
                "reply_text": "re: suivi de dossier",
            },
        ),
    ],
)
def test_transform_debug_mode(df_emails, expected_result, expected_debug_info):
    """Unit test of the debug mode."""

    # Copy for later load/save test
    df_copy = df_emails.copy()

    # Instanciate manually a detector
    detector = ReplyDetector(
        name="reply",
        header_column="clean_header",
    )

    # Get column names
    res_col = detector.result_column
    debug_dict_col = detector.debug_dict_col

    # Transform data
    df_emails.debug = True
    df_emails = detector.transform(df_emails)

    # Collect results
    result = df_emails[res_col].iloc[0]
    debug_result = df_emails[debug_dict_col].iloc[0]

    # Test result
    assert result == expected_result
    assert debug_result == expected_debug_info


@pytest.mark.parametrize(
    "df, expected_result",
    [
        (
            pd.DataFrame(
                {
                    "from": ["test@gmail.com"],
                    "header": ["Re: Suivi de dossier"],
                    "body": ["Bonjour,\nle traitement de ma demande est deplorable.\nje suis tres en colere.\n"],
                }
            ),
            True,
        ),
        (
            pd.DataFrame(
                {
                    "from": ["test@gmail.com"],
                    "header": ["re: Envoi d'un document de la Société Imaginaire"],
                    "body": ["Bonjour,\nLe traitement de ma demande est déplorable.\nJe suis très en colère.\n"],
                }
            ),
            True,
        ),
        (
            pd.DataFrame(
                {
                    "from": ["test@gmail.com"],
                    "header": ["te: Virement"],
                    "body": [
                        "Bonjour,\nJe vous confirme l'annulation du rdv du 01/01/2022 "
                        + "à 16h.\nBien cordialement,\nJohn Smith."
                    ],
                }
            ),
            False,
        ),
        (
            pd.DataFrame(
                {
                    "from": ["test@gmail.com"],
                    "header": [""],
                    "body": [
                        "Bonjour,\nJe vous confirme l'annulation du rdv du 01/01/2022 "
                        + "à 16h.\nBien cordialement,\nJohn Smith."
                    ],
                }
            ),
            False,
        ),
    ],
)
def test_pipeline_from_config(df, expected_result):
    """
    Instanciate from a config and test the pipeline.
    """
    # Pipeline config key
    pipeline_key = "reply_pipeline"

    # Create pipeline from config
    pipeline = MelusinePipeline.from_config(config_key=pipeline_key)

    # Apply pipeline on data
    df_transformed = pipeline.transform(df)
    result = df_transformed["reply_result"][0]

    # Check
    assert result == expected_result
