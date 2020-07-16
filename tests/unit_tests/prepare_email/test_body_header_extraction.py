import pandas as pd
import numpy as np
from melusine.prepare_email.body_header_extraction import extract_last_body
from melusine.prepare_email.body_header_extraction import extract_body
from melusine.prepare_email.body_header_extraction import extract_header


structured_body = [
    {
        "meta": {"date": None, "from": None, "to": None},
        "structured_text": {
            "header": "demande document",
            "text": [
                {"part": "Bonjour. ", "tags": "HELLO"},
                {"part": "Je vous remercie pour le document", "tags": "BODY"},
                {"part": "Cordialement,", "tags": "GREETINGS"},
                {"part": "Mr Unknown", "tags": "BODY"},
            ],
        },
    },
    {
        "meta": {
            "date": " mar. 22 mai 2018 à 10:20",
            "from": "  <destinataire@gmail.fr> ",
            "to": None,
        },
        "structured_text": {
            "header": "demande document",
            "text": [
                {"part": "Bonjour. ", "tags": "HELLO"},
                {
                    "part": "Merci de bien vouloir prendre connaissance du document ci-joint",
                    "tags": "BODY",
                },
                {"part": "Cordialement,", "tags": "GREETINGS"},
                {"part": "Votre mutuelle", "tags": "BODY"},
                {
                    "part": "La visualisation des fichiers PDF nécessite Adobe Reader.",
                    "tags": "FOOTER",
                },
            ],
        },
    },
]


def test_extract_last_body():
    input_df = pd.DataFrame({"structured_body": [structured_body]})

    output_df = pd.Series(["Je vous remercie pour le document "])

    result = input_df.apply(extract_last_body, axis=1)
    pd.testing.assert_series_equal(result, output_df)


message_dict = {
    "meta": {
        "date": " mar. 22 mai 2018 à 10:20",
        "from": "  <destinataire@gmail.fr> ",
        "to": None,
    },
    "structured_text": {
        "header": "demande document",
        "text": [
            {"part": "Bonjour. ", "tags": "HELLO"},
            {
                "part": "Merci de bien vouloir prendre connaissance du document ci-joint",
                "tags": "BODY",
            },
            {"part": "Cordialement,", "tags": "GREETINGS"},
            {"part": "Votre mutuelle", "tags": "BODY"},
            {
                "part": "La visualisation des fichiers PDF nécessite Adobe Reader.",
                "tags": "FOOTER",
            },
        ],
    },
}


def test_extract_body():
    input_dict = message_dict

    output = "Merci de bien vouloir prendre connaissance du document ci-joint "

    result = extract_body(input_dict)
    np.testing.assert_string_equal(result, output)


def test_extract_header():
    input_dict = message_dict

    output = "demande document"

    result = extract_header(input_dict)
    np.testing.assert_string_equal(result, output)
