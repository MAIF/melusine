"""
Unit tests of the TransferDetector.
"""


import pandas as pd
import pytest
from pandas import DataFrame

from melusine.detectors import TransferDetector
from melusine.message import Message
from melusine.pipeline import MelusinePipeline


def test_instanciation():
    """Instanciation base test."""

    detector = TransferDetector(name="transfer", header_column="det_clean_header", messages_column="messages")
    assert isinstance(detector, TransferDetector)


@pytest.mark.parametrize(
    "row, good_result",
    [
        (
            {
                "reply_text": "tr: Devis habitation",
                "messages": [
                    Message(
                        meta="",
                        text="Bonjour, je vous confirme l'annulation du rdv du 01/01/2022 "
                        + "à 16h. Bien cordialement, John Smith.",
                        tags=[
                            ("HELLO", "Bonjour,"),
                            (
                                "BODY",
                                "je vous confirme l'annulation du rdv du 01/01/2022 " + "à 16h. ",
                            ),
                            ("GREETINGS", "Cordialement, John Smith."),
                        ],
                    )
                ],
            },
            True,
        ),
        (
            {
                "reply_text": "re: Envoi d'un document de la Société Imaginaire",
                "messages": [
                    Message(
                        meta="this is meta",
                        text="Bonjour, je vous confirme l'annulation du rdv du 01/01/2022 "
                        + "à 16h. Bien cordialement, John Smith.",
                        tags=[
                            ("HELLO", "Bonjour,"),
                            (
                                "BODY",
                                "je vous confirme l'annulation du rdv du 01/01/2022 " + "à 16h. ",
                            ),
                            ("GREETINGS", "Cordialement, John Smith."),
                        ],
                    )
                ],
            },
            True,
        ),
        (
            {
                "reply_text": "re: Virement",
                "messages": [
                    Message(
                        meta="",
                        text="Bonjour, je vous confirme l'annulation du rdv du 01/01/2022 "
                        + "à 16h. Bien cordialement, John Smith.",
                        tags=[
                            ("HELLO", "Bonjour,"),
                            (
                                "BODY",
                                "je vous confirme l'annulation du rdv du 01/01/2022 " + "à 16h. ",
                            ),
                            ("GREETINGS", "Cordialement, John Smith."),
                        ],
                    )
                ],
            },
            False,
        ),
        (
            {
                "reply_text": "",
                "messages": [
                    Message(
                        meta="",
                        text="Bonjour, je vous confirme l'annulation du rdv du 01/01/2022 "
                        + "à 16h. Bien cordialement, John Smith.",
                        tags=[
                            ("HELLO", "Bonjour,"),
                            (
                                "BODY",
                                "je vous confirme l'annulation du rdv du 01/01/2022 " + "à 16h. ",
                            ),
                            ("GREETINGS", "Cordialement, John Smith."),
                        ],
                    )
                ],
            },
            False,
        ),
    ],
)
def test_deterministic_detect(row, good_result):
    """Method base test."""

    # Instanciate manually a detector
    detector = TransferDetector(
        name="transfer",
        header_column="det_clean_header",
        messages_column="messages",
    )
    row = detector.detect(row)
    res = row[detector.result_column]
    assert res == good_result


@pytest.mark.parametrize(
    "df_emails, expected_result",
    [
        (
            DataFrame(
                {
                    "det_clean_header": "tr: Rdv",
                    "messages": [
                        [
                            Message(
                                meta="",
                                text="Bonjour, je vous confirme l'annulation du rdv du 01/01/2022 "
                                + "à 16h. Bien cordialement, John Smith.",
                                tags=[
                                    ("HELLO", "Bonjour,"),
                                    (
                                        "BODY",
                                        "je vous confirme l'annulation du rdv du 01/01/2022 " + "à 16h. ",
                                    ),
                                    ("GREETINGS", "Cordialement, John Smith."),
                                ],
                            )
                        ]
                    ],
                }
            ),
            True,
        ),
    ],
)
def test_transform(df_emails, expected_result):
    """Unit test of the transform() method."""

    # Copy for later load/save test
    df_copy = df_emails.copy()

    # Instanciate manually a detector
    detector = TransferDetector(
        name="transfer",
        header_column="det_clean_header",
        messages_column="messages",
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
                    "det_clean_header": ["Tr: Suivi de dossier"],
                    "messages": [
                        [
                            Message(
                                meta="",
                                text="Bonjour, je vous confirme l'annulation du rdv du 01/01/2022 "
                                + "à 16h. Bien cordialement, John Smith.",
                                tags=[
                                    ("HELLO", "Bonjour,"),
                                    (
                                        "BODY",
                                        "je vous confirme l'annulation du rdv du 01/01/2022 " + "à 16h. ",
                                    ),
                                    ("GREETINGS", "Cordialement, John Smith."),
                                ],
                            )
                        ]
                    ],
                }
            ),
            True,
            {
                "reply_text": "tr: suivi de dossier",
                "messages[0].meta": "",
                "TransferRegex": {
                    "match_result": True,
                    "negative_match_data": {},
                    "neutral_match_data": {},
                    "positive_match_data": {"DEFAULT": [{"match_text": "tr:", "start": 0, "stop": 3}]},
                },
            },
        ),
    ],
)
def test_transform_debug_mode(df_emails, expected_result, expected_debug_info):
    """Unit test of the debug mode."""

    # Copy for later load/save test
    df_copy = df_emails.copy()

    # Instanciate manually a detector
    detector = TransferDetector(
        name="transfer",
        header_column="det_clean_header",
        messages_column="messages",
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
                    "header": ["tr :Suivi de dossier"],
                    "body": [
                        "",
                    ],
                }
            ),
            True,
        ),
        (
            pd.DataFrame(
                {
                    "from": ["test@gmail.com"],
                    "header": ["fwd: Envoi d'un document de la Société Imaginaire"],
                    "body": [
                        "Bonjour,\n\n\n\n\n\nUn taux d’humidité de 30% a été relevé le 01/01/2024.\n\n\n\nNous reprendrons contact avec l’assurée"
                        + " en Aout 2022.\n\n\n\n\n\n\nBien cordialement,\n\n\n\n\n\nNuméro Auxiliaire : 000000 A / 000000 B\n\n\n\n\n\n\n\n\n"
                        + "SMITH Kim\n\n\nTEST\n-\n\nL\na\nV\nalorisation du\nP\natrimoine\n\n\n2, rue du Test\n\n\n00000 NIORT"
                        + "\n\n\n\n\n\n\n\nTél :    0143740992\n\n\nPort :  0767396737\n\n\nhttp://lvpfrance.fr\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
                        + "\n\n\nDe :\nAccueil - Alex Dupond <accueil@test.fr>\n\n\n\nEnvoyé :\njeudi 01 janvier 2024 01:01\n\n\nÀ "
                        + ":\nCommercial <etudes@test.fr>\n\n\nObjet :\nTR: Evt : A0000000B survenu le 01/01/2024 - Intervention entreprise"
                        + " partenaire\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nDe :\n\n\ntest@maif.fr\n[\nmailto:test@maif.fr\n]\n\n\n\n"
                        + "Envoyé :\njeudi 01 janvier 2024 01:01\n\n\nÀ :\nAccueil - Alex Dupond\n\n\nObjet :\nEvt : A0000000B survenu le 01/01/2024"
                        + " - Intervention entreprise partenaire\n\n\n\n\n\nMerci de bien vouloir prendre connaissance du document ci-joint.",
                    ],
                }
            ),
            True,
        ),
        (
            pd.DataFrame(
                {
                    "from": ["test@gmail.com"],
                    "header": ["Virement"],
                    "body": [
                        "Bonjour,\n\n\n\n\n\nUn taux d’humidité de 30% a été relevé le 01/01/2024.\n\n\n\nNous reprendrons contact avec l’assurée"
                        + " en Aout 2022.\n\n\n\n\n\n\nBien cordialement,\n\n\n\n\n\nNuméro Auxiliaire : 000000 A / 000000 B\n\n\n\n\n\n\n\n\n"
                        + "SMITH Kim\n\n\nTEST\n-\n\nL\na\nV\nalorisation du\nP\natrimoine\n\n\n2, rue du Test\n\n\n00000 NIORT"
                        + "\n\n\n\n\n\n\n\nTél :    0143740992\n\n\nPort :  0767396737\n\n\nhttp://lvpfrance.fr\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
                        + "\n\n\nDe :\nAccueil - Alex Dupond <accueil@test.fr>\n\n\n\nEnvoyé :\njeudi 01 janvier 2024 01:01\n\n\nÀ "
                        + ":\nCommercial <etudes@test.fr>\n\n\nObjet :\nTR: Evt : A0000000B survenu le 01/01/2024 - Intervention entreprise"
                        + " partenaire\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nDe :\n\n\ntest@maif.fr\n[\nmailto:test@maif.fr\n]\n\n\n\n"
                        + "Envoyé :\njeudi 01 janvier 2024 01:01\n\n\nÀ :\nAccueil - Alex Dupond\n\n\nObjet :\nEvt : A0000000B survenu le 01/01/2024"
                        + " - Intervention entreprise partenaire\n\n\n\n\n\nMerci de bien vouloir prendre connaissance du document ci-joint.",
                    ],
                }
            ),
            False,
        ),
        (
            pd.DataFrame(
                {
                    "from": ["test@gmail.com"],
                    "header": ["tr: virement"],
                    "body": [
                        "Bonjour,\n\n\n\n\n\nUn taux d’humidité de 30% a été relevé le 01/01/2001.\n\n\n\nNous reprendrons contact avec l’assurée"
                        + " en Aout 2022.\n\n\n\n\n\n\nBien cordialement,\n\n\n\n\n\nNuméro : 000000\n\n\n\n\n\n\n\n\nJohn"
                        + " Smith\n\n\nL\na\nValorisation du\nPatrimoine\n\n\n1, rue de la Paix\n\n\n79000 Niort"
                        + "\n\n\n\n\n\n\n\nTél :    0123456789\n\n\nPort :  0123456789\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
                        + "\n\n\nDe :\nAccueil - John Smith <john.smith@test.fr>\n\n\n\nEnvoyé :\njeudi 01 janvier 2001 01:01\n\n\nÀ "
                        + ":\nCommercial <test@test.fr>\n\n\nObjet :\nTR: Accident survenu le 01/01/2021 - Intervention"
                        + " partenaire\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nDe :\n\n\ntest@test.fr\n[\nmailto:test@test.fr\n]\n\n\n\n"
                        + "Envoyé :\njeudi 01 janvier 2001 01:01\n\n\nÀ :\nAccueil\n\n\nObjet :\nAccident survenu le 01/01/2001"
                        + " - Intervention partenaire\n\n\n\n\n\nMerci de bien vouloir prendre connaissance du document ci-joint.\n\n\n\n"
                        + "Cordialement\n",
                    ],
                }
            ),
            True,
        ),
    ],
)
def test_pipeline_from_config(df, expected_result):
    """
    Instanciate from a config and test the pipeline.
    """
    # Pipeline config key
    pipeline_key = "transfer_pipeline"

    # Create pipeline from config
    pipeline = MelusinePipeline.from_config(config_key=pipeline_key)

    # Apply pipeline on data
    df_transformed = pipeline.transform(df)
    result = df_transformed["transfer_result"][0]

    # Check
    assert result == expected_result
