import pandas as pd
from melusine.prepare_email.mail_segmenting import structure_email, tag_signature

structured_historic = [
    {
        "text": " \n  \n  \n Bonjours, \n  \n Suite a notre conversation \
téléphonique de Mardi , pourriez vous me dire la \n somme que je vous \
dois afin d'd'être en régularisation . \n  \n Merci bonne journée",
        "meta": "",
    },
    {
        "text": " \n Bonjour. \n  \n Merci de bien vouloir prendre connaissance \
du document ci-joint : \n 1 - Relevé d'identité postal MUTUELLE \
(contrats) \n  \n Sentiments mutualistes. \n  \n La Mutuelle \n  \n \
La visualisation des fichiers PDF nécessite Adobe Reader. \n  ",
        "meta": " \n  \n Le mar. 22 mai 2018 à 10:20,  \
<gestionsocietaire@mutuelle.fr> a écrit\xa0:",
    },
]

output = [
    {
        "meta": {"date": None, "from": None, "to": None},
        "structured_text": {
            "header": None,
            "text": [
                {"part": "Bonjours,", "tags": "HELLO"},
                {
                    "part": " Suite a notre conversation \
téléphonique de Mardi , pourriez vous me dire la somme que je vous dois \
afin d'd'être en régularisation .",
                    "tags": "BODY",
                },
                {"part": "Merci bonne journée", "tags": "GREETINGS"},
            ],
        },
    },
    {
        "meta": {
            "date": " mar. 22 mai 2018 à 10:20",
            "from": "  <gestionsocietaire@mutuelle.fr> ",
            "to": None,
        },
        "structured_text": {
            "header": None,
            "text": [
                {"part": "Bonjour.", "tags": "HELLO"},
                {
                    "part": "Merci de bien vouloir prendre \
connaissance du document ci-joint : 1 - Relevé d'identité postal MUTUELLE \
(contrats) ",
                    "tags": "BODY",
                },
                {"part": "Sentiments mutualistes.", "tags": "GREETINGS"},
                {"part": "La Mutuelle ", "tags": "BODY"},
                {
                    "part": "La visualisation des fichiers \
PDF nécessite Adobe Reader.",
                    "tags": "FOOTER",
                },
            ],
        },
    },
]


def test_structure_email():
    input_df = pd.DataFrame({"structured_historic": [structured_historic]})

    output_df = pd.Series([output])

    result = input_df.apply(structure_email, axis=1)
    pd.testing.assert_series_equal(result, output_df)


structured_historic_signature = [
    {
        "text": " \n  \n  \n Bonjours, \n  \n Suite a notre conversation \
téléphonique de Mardi , pourriez vous me dire la \n somme que je vous \
dois afin d'd'être en régularisation . \n  \n Merci bonne journée\nJean Dupont",
        "meta": "",
    },
    {
        "text": " \n Bonjour. \n  \n Merci de bien vouloir prendre connaissance \
du document ci-joint : \n 1 - Relevé d'identité postal MUTUELLE \
(contrats) \n  \n Sentiments mutualistes. \n  \n La Mutuelle \n  \n \
La visualisation des fichiers PDF nécessite Adobe Reader. \n  ",
        "meta": " \n  \n Le mar. 22 mai 2018 à 10:20,  \
<gestionsocietaire@mutuelle.fr> a écrit\xa0:",
    },
]

output_signature = [
    {
        "meta": {"date": None, "from": None, "to": None},
        "structured_text": {
            "header": None,
            "text": [
                {"part": "Bonjours,", "tags": "HELLO"},
                {
                    "part": " Suite a notre conversation \
téléphonique de Mardi , pourriez vous me dire la somme que je vous dois \
afin d'd'être en régularisation .",
                    "tags": "BODY",
                },
                {"part": "Merci bonne journée", "tags": "GREETINGS"},
                {"part": "Jean Dupont", "tags": "SIGNATURE"},
            ],
        },
    },
    {
        "meta": {
            "date": " mar. 22 mai 2018 à 10:20",
            "from": "  <gestionsocietaire@mutuelle.fr> ",
            "to": None,
        },
        "structured_text": {
            "header": None,
            "text": [
                {"part": "Bonjour.", "tags": "HELLO"},
                {
                    "part": "Merci de bien vouloir prendre \
connaissance du document ci-joint : 1 - Relevé d'identité postal MUTUELLE \
(contrats) ",
                    "tags": "BODY",
                },
                {"part": "Sentiments mutualistes.", "tags": "GREETINGS"},
                {"part": "La Mutuelle ", "tags": "BODY"},
                {
                    "part": "La visualisation des fichiers PDF nécessite Adobe Reader.",
                    "tags": "FOOTER",
                },
            ],
        },
    },
]


def test_tag_signature():
    input_df = pd.DataFrame({"structured_historic": [structured_historic_signature]})

    output_df = pd.Series([output_signature])
    input_df["structured_body"] = input_df.apply(structure_email, axis=1)
    result = input_df.apply(tag_signature, axis=1)
    pd.testing.assert_series_equal(result, output_df)
