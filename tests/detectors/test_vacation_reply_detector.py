"""
Unit tests of the VacationReplyDetector
"""
import pandas as pd
import pytest
from pandas import DataFrame

from melusine.detectors import VacationReplyDetector
from melusine.message import Message
from melusine.pipeline import MelusinePipeline


def test_instanciation():
    """Instanciation base test."""
    detector = VacationReplyDetector(
        name="vacation_reply",
        messages_column="messages",
    )
    assert isinstance(detector, VacationReplyDetector)


@pytest.mark.parametrize(
    "df, good_result",
    [
        (
            DataFrame(
                {
                    "messages": [
                        [
                            Message(
                                text="Bonjour, je vous confirme l'annulation du rdv du 01/01/2022 "
                                + "à 16h. Bien cordialement, John Smith.",
                                tags=[
                                    ("HELLO", "Bonjour,"),
                                    (
                                        "BODY",
                                        "je vous confirme l'annulation du rdv du 01/01/2022 à 16h.",
                                    ),
                                    ("GREETINGS", "Bien cordialement, John Smith."),
                                ],
                            )
                        ]
                    ]
                }
            ),
            False,
        ),
        (
            DataFrame(
                {
                    "messages": [
                        [
                            Message(
                                text="Bonjour, \nActuellement en conge je prendrai connaissance"
                                + " de votre message ulterieurement.\nCordialement,",
                                tags=[
                                    ("HELLO", "Bonjour,"),
                                    (
                                        "BODY",
                                        "Actuellement en conge je prendrai connaissance de votre message ulterieurement.",
                                    ),
                                    ("GREETINGS", "Cordialement, "),
                                ],
                            )
                        ]
                    ]
                }
            ),
            True,
        ),
    ],
)
def test_transform(df, good_result):
    """Unit test of the transform() method."""
    df_copy = df.copy()

    message_column = "messages"

    detector = VacationReplyDetector(
        name="vacation_reply",
        messages_column=message_column,
    )
    output_col = detector.result_column

    df = detector.transform(df)
    result = df[output_col][0]
    assert result == good_result


@pytest.mark.parametrize(
    "df, good_detection_result, good_debug_info",
    [
        (
            DataFrame(
                {
                    "messages": [
                        [
                            Message(
                                text="Bonjour, \nActuellement en conge je prendrai connaissance"
                                + " de votre message ulterieurement.\nCordialement,",
                                tags=[
                                    ("HELLO", "Bonjour,"),
                                    (
                                        "BODY",
                                        "Actuellement en conge je prendrai connaissance de votre message ulterieurement.",
                                    ),
                                    ("GREETINGS", "Cordialement, "),
                                ],
                            )
                        ]
                    ]
                }
            ),
            True,
            {
                "parts": [
                    (
                        "BODY",
                        "Actuellement en conge je prendrai connaissance de votre message ulterieurement.",
                    )
                ],
                "text": "Actuellement en conge je prendrai connaissance de votre message ulterieurement.",
                "VacationReplyRegex": {
                    "match_result": True,
                    "negative_match_data": {},
                    "neutral_match_data": {},
                    "positive_match_data": {
                        "VAC_REP_HOLIDAYS": [{"match_text": "Actuellement " "en " "conge", "start": 0, "stop": 21}],
                        "VAC_REP_OUT_OF_OFFICE": [
                            {"match_text": "je " "prendrai " "connaissance", "start": 22, "stop": 46}
                        ],
                    },
                },
            },
        ),
    ],
)
def test_transform_debug_mode(df, good_detection_result, good_debug_info):
    """Unit test of the debug mode."""
    df_copy = df.copy()

    messages_column = "messages"

    detector = VacationReplyDetector(
        name="vacation_reply",
        messages_column=messages_column,
    )
    output_col = detector.result_column
    debug_dict_col = detector.debug_dict_col

    # Transform data
    df.debug = True
    df = detector.transform(df)

    # Collect results
    result = df[output_col].iloc[0]
    debug_result = df[debug_dict_col].iloc[0]

    # Test result
    assert result == good_detection_result
    assert debug_result == good_debug_info


@pytest.mark.parametrize(
    "df, expected_result",
    [
        (
            pd.DataFrame(
                {
                    "from": ["test@gmail.com"],
                    "header": [""],
                    "body": [
                        "Bonjour, \nActuellement en congé je prendrai connaissance"
                        + " de votre message ultérieurement.\nCordialement,"
                    ],
                }
            ),
            True,
        ),
        (
            pd.DataFrame(
                {
                    "from": ["test@gmail.com"],
                    "header": [""],
                    "body": [
                        "Bonjour,\nje vous confirme l'annulation du rdv du 01/01/2001 "
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
    pipeline_key = "vacation_reply_pipeline"

    # Create pipeline from config
    pipeline = MelusinePipeline.from_config(config_key=pipeline_key)

    # Apply pipeline on data
    df_transformed = pipeline.transform(df)
    result = df_transformed["vacation_reply_result"][0]

    # Check
    assert result == expected_result
