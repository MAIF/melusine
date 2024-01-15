"""
================================== Test-Case fixtures ==================================
# Overview
Tests-cases are defined in this file and tested using `test_emails_generic`.
The fixture `test_message` will successively take the value of every test-case defined
below.

# Adding a new test case
To add a new test-case, create below a variable named "testcase_${NAME}".
The variable should be a dict with the following fields:
- Input fields (ex: "body", "header", "from", "to", etc):
  Used to create the input email passed to the pipeline
-- Example --
testcase_hello_world = {"body": "Hello World"}

- Expected output fields (ex: normalizer_expected, segmenter_expected, etc):
  Used to verify the data resulting from a transformation
  The `segmenter_expected` field will test the results posterior to the pipeline step
  named `segmenter`.
  The content of an expected output field is a dictionary with a key for each DataFrame
  columns to be tested.
-- Example --
testcase_hello_world = {
  "body": "Hello World",
  "tokenizer_expected: {
    "tokens": ['hello', 'world']"   # <== Test the content of the tokens column
    }
}

# Test Message class attributes:
The `messages` column contains a list of Message class instances.
The value of the class instance attributes can be tested
with the syntax `messages.attribute`.
-- Example --
testcase_hello_world = {
  "body": "Hello World\nMessage transféré\nBonjour Monde",
  "segmenter_expected: {
    "messages.text": ['Hello World', 'Bonjour Monde']"   # <== Test the text attribute
    "messages.meta": [None, 'Message transfere']"   # <== Test the meta attribute
    }
}
========================================================================================
"""
import pytest

testcase_initial_cleaning_1 = dict(
    test_name="Initial leaning line breaks",
    body="BonJour wORLD\r\n L'orem \r\n\t\r\nIp-sum\r  Lo_rem  \nip.sum.",
    body_cleaner_expected={
        "tmp_clean_body": "BonJour wORLD\nL'orem\nIp-sum\nLo_rem\nip.sum.",
    },
    content_tagger_expected={
        "messages.tags": [
            [
                ("HELLO", "BonJour wORLD"),
                ("BODY", "L'orem"),
                ("BODY", "Ip-sum"),
                ("BODY", "Lo_rem"),
                ("BODY", "ip.sum."),
            ],
        ],
    },
    tokenizer_expected={
        "ml_body_tokens": [
            "l",
            "orem",
            "ip-sum",
            "lo_rem",
            "ip",
            "sum",
        ],
    },
)

# Watch-out : Multi-line string are NOT Tuples ("abc" "def") != ("abc", "def")
testcase_initial_cleaning_2 = dict(
    test_name="Initial leaning special characters",
    body="Hello\xa0World\n’œ’    <\nyoo\n>",
    body_cleaner_expected={
        "tmp_clean_body": "Hello World\n'oe' <yoo>",
    },
)


# Watch-out : Multi-line strings are NOT Tuples ("abc" "def") != ("abc", "def")
testcase_segmentation1 = dict(
    test_name="Segmentation Date/Envoyé/À/Objet/Le",
    body=(
        "De : test@free.fr <test@free.fr> \nEnvoyé : lundi 30 août 2021 21:26 \nÀ : _Délégation 00.0 - "
        "Ville <ville@maif.fr> \nObjet : Re: Soc : 0000000P - Votre lettre \nBonjour, \n"
        "Vous trouverez ci-joint l'attestation \nMerci de me confirmer la bonne réception de ce "
        "message. \nVous en remerciant par avance. \nCordialement, \nJean Dupont \nLe 2021-08-18 10:30, "
        "ville@malf.fr a écrit : \nBonjour, \nVeuillez trouver ci-jointe la lettre \nLa visualisation des "
        "fichiers PDF nécessite Adobe Reader. \nSentiments mutualistes. \nLa MAIF \n"
    ),
    segmenter_expected={
        "messages.text": [
            (
                "Bonjour,\nVous trouverez ci-joint l'attestation\nMerci de me confirmer la bonne réception "
                "de ce message.\nVous en remerciant par avance.\nCordialement,\nJean Dupont"
            ),
            (
                "Bonjour,\nVeuillez trouver ci-jointe la lettre\nLa visualisation des fichiers PDF nécessite "
                "Adobe Reader.\nSentiments mutualistes.\nLa MAIF"
            ),
        ],
        "messages.meta": [
            (
                "De : test@free.fr <test@free.fr>\nEnvoyé : lundi 30 août 2021 21:26\nÀ : _Délégation 00.0 - "
                "Ville <ville@maif.fr>\nObjet : Re: Soc : 0000000P - Votre lettre"
            ),
            "Le 2021-08-18 10:30, ville@malf.fr a écrit :",
        ],
    },
    content_tagger_expected={
        "messages.tags": [
            [
                ("HELLO", "Bonjour,"),
                ("BODY", "Vous trouverez ci-joint l'attestation"),
                ("BODY", "Merci de me confirmer la bonne réception de ce message."),
                ("THANKS", "Vous en remerciant par avance."),
                ("GREETINGS", "Cordialement,"),
                ("SIGNATURE_NAME", "Jean Dupont"),
            ],
            [
                ("HELLO", "Bonjour,"),
                ("BODY", "Veuillez trouver ci-jointe la lettre"),
                ("FOOTER", "La visualisation des fichiers PDF nécessite Adobe Reader."),
                ("GREETINGS", "Sentiments mutualistes."),
                ("SIGNATURE_NAME", "La MAIF"),
            ],
        ],
    },
)

# Watch-out : Multi-line string are NOT Tuples ("abc" "def") != ("abc", "def")
testcase_segmentation2 = dict(
    test_name="Segmentation Direct Transfer",
    body=(
        "De : Jean Dupond <jean.dupond.971@gmail.com> \nEnvoyé : mardi 31 août 2021 21:45 \n"
        "À : _Délégation Conseil 000 - La Ville <la.ville000@maif.fr> \n"
        "Objet : Demande d'attestation Identifiant : 0000000N \nBonjour \n"
        "Pouvez-vous me transmettre deux attestations au nom de mes enfants \n- Jane Dupond \n- Joe Dupond \n"
        "Merci par avance \n-- \nCordialement \nMr Jean Dupond"
    ),
    segmenter_expected={
        "messages.text": [
            (
                "Bonjour\nPouvez-vous me transmettre deux attestations au nom de mes enfants\n"
                "- Jane Dupond\n- Joe Dupond\nMerci par avance\n--\nCordialement\nMr Jean Dupond"
            )
        ],
        "messages.meta": [
            (
                "De : Jean Dupond <jean.dupond.971@gmail.com>\nEnvoyé : mardi 31 août 2021 21:45\n"
                "À : _Délégation Conseil 000 - La Ville <la.ville000@maif.fr>\n"
                "Objet : Demande d'attestation Identifiant : 0000000N"
            ),
        ],
    },
    content_tagger_expected={
        "messages.tags": [
            [
                ("HELLO", "Bonjour"),
                (
                    "BODY",
                    "Pouvez-vous me transmettre deux attestations au nom de mes enfants",
                ),
                ("BODY", "- Jane Dupond"),
                ("BODY", "- Joe Dupond"),
                ("THANKS", "Merci par avance"),
                ("GREETINGS", "Cordialement"),
                ("SIGNATURE_NAME", "Mr Jean Dupond"),
            ]
        ],
    },
)

# Watch-out : Multi-line string are NOT Tuples ("abc" "def") != ("abc", "def")
testcase_segmentation3 = dict(
    test_name="Segmentation Début du message transféré",
    body=(
        "Envoyé de mon iPhone \nDébut du message transféré : \nDe: Jane Dupond <jane.dupond@hotmail.fr> \n"
        "Date: 11 août 2021 à 17:04:01 UTC+2 \nÀ: Joe DUPOND <joe@du-pond.fr> \n"
        "Objet: Rép. : X - ETAT DES LIEUX \nBonjour Mme X, \n"
        "Suite à l'état des lieux de ce matin, je suis passé à l'agence transmettre les objets \n"
        "Bien cordialement \nJane Dupond \nEnvoyé de mon iPhone \nLe 11 août 2021à 15:35, "
        "Joe DUPOND <jdupond@gmail.fr> a écrit : \nBonjour, \n"
        "Veuillez trouver ci-joint votre état des lieux sortant. \n"
        "Vous en souhaitant bonne réceptionn, \nBien cordialement, \nJoe DUPOND \n "
    ),
    segmenter_expected={
        "messages.text": [
            "Envoyé de mon iPhone",
            (
                "Bonjour Mme X,\nSuite à l'état des lieux de ce matin, je suis passé à l'agence transmettre "
                "les objets\nBien cordialement\nJane Dupond\nEnvoyé de mon iPhone"
            ),
            (
                "Bonjour,\nVeuillez trouver ci-joint votre état des lieux sortant.\n"
                "Vous en souhaitant bonne réceptionn,\nBien cordialement,\nJoe DUPOND"
            ),
        ],
        "messages.meta": [
            "",
            (
                "Début du message transféré :\nDe: Jane Dupond <jane.dupond@hotmail.fr>\n"
                "Date: 11 août 2021 à 17:04:01 UTC+2\nÀ: Joe DUPOND <joe@du-pond.fr>\n"
                "Objet: Rép. : X - ETAT DES LIEUX"
            ),
            "Le 11 août 2021à 15:35, Joe DUPOND <jdupond@gmail.fr> a écrit :",
        ],
    },
)

# Watch-out : Multi-line string are NOT Tuples ("abc" "def") != ("abc", "def")
testcase_segmentation4 = dict(
    test_name="Segmentation Original Message",
    body=(
        "De : marie@protonmail.com<marie@protonmail.com> \nEnvoyé : mardi 31 août 202114:04 \n"
        "À : _Délégation 00.1- Ville <ville@maif.fr> \n"
        "Objet : Re : Soc : 0000000M - Votre attestation Assurance Habitation Responsabilité civile locative \n"
        "Bonjour, \nJe vous renvoie mon RIB concernant le contrat \nd'assurance Habitation Responsabilité civile "
        "locative. \nBien à vous \nMarie \nN° sociétaire : 0000000M \nSent with Proto_nMail Secure Email. \n"
        "--- Original Message --- \nLe mardi 31 août 2021 à 11:09, <ville@maif.fr> a écrit : \n"
        "Bonjour, \nVeuillez trouver ci-joint l'attestation « Responsabilité civile locative » \n"
        "que vous nous avez demandée. \nLa visualisation des fichiers PDF nécessite Adobe Reader. \n"
        "Sentiments mutualistes. \nLa MAIF \n"
    ),
    segmenter_expected={
        "messages.text": [
            (
                "Bonjour,\nJe vous renvoie mon RIB concernant le contrat\nd'assurance Habitation "
                "Responsabilité civile locative.\nBien à vous\nMarie\nN° sociétaire : 0000000M\n"
                "Sent with Proto_nMail Secure Email."
            ),
            (
                "Bonjour,\nVeuillez trouver ci-joint l'attestation « Responsabilité civile locative »\n"
                "que vous nous avez demandée.\nLa visualisation des fichiers PDF nécessite Adobe Reader.\n"
                "Sentiments mutualistes.\nLa MAIF"
            ),
        ],
        "messages.meta": [
            (
                "De : marie@protonmail.com<marie@protonmail.com>\nEnvoyé : mardi 31 août 202114:04\n"
                "À : _Délégation 00.1- Ville <ville@maif.fr>\nObjet : Re : Soc : 0000000M - Votre attestation "
                "Assurance Habitation Responsabilité civile locative"
            ),
            "Original Message ---\nLe mardi 31 août 2021 à 11:09, <ville@maif.fr> a écrit :",
        ],
    },
)


# Watch-out : Multi-line string are NOT Tuples ("abc" "def") != ("abc", "def")
testcase_segmentation5 = dict(
    test_name="Segmentation Le lun. xxx a écrit",
    body=(
        "Bonjour, \nVeuillez trouver en PJ mon RI \n"
        "Le lun. 30 août 2021à 09:40, DUPOND Marie <marie@maif.fr> a écrit : \nBonjour"
    ),
    segmenter_expected={
        "messages.text": ["Bonjour,\nVeuillez trouver en PJ mon RI", "Bonjour"],
        "messages.meta": [
            "",
            "Le lun. 30 août 2021à 09:40, DUPOND Marie <marie@maif.fr> a écrit :",
        ],
    },
)

# Watch-out : Multi-line string are NOT Tuples ("abc" "def") != ("abc", "def")
testcase_segmentation6 = dict(
    test_name="Segmentation Direct Transfer",
    body="Bonjour et merci\nCordialement",
    content_tagger_expected={
        "messages.tags": [
            [
                ("THANKS", "Bonjour et merci"),
                ("GREETINGS", "Cordialement"),
            ]
        ],
    },
)

# Watch-out : Multi-line string are NOT Tuples ("abc" "def") != ("abc", "def")
testcase_transfer_1 = dict(
    test_name="Direct transfer",
    body="De:\nsender.before.transfer@test.fr\nDate:\n3 mars 2023 à 16:42:50\nBonjour\nbla bla",
    transferred_email_processor_expected={
        "messages.text": ["Bonjour\nbla bla"],
        "messages.meta": ["De:\nsender.before.transfer@test.fr\nDate:\n3 mars 2023 à 16:42:50"],
        "det_original_from": "sender.before.transfer@test.fr",
    },
)

testcase_transfer_2 = dict(
    test_name="FOOTER + Transfer",
    body="Envoyé de mon iphone\nDe:\nsender.before.transfer@test.fr\nDate:\n3 mars 2023 à 16:42:50\nBonjour\nbla bla",
    transferred_email_processor_expected={
        "messages.text": ["Bonjour\nbla bla"],
        "messages.meta": ["De:\nsender.before.transfer@test.fr\nDate:\n3 mars 2023 à 16:42:50"],
        "det_original_from": "sender.before.transfer@test.fr",
    },
)

testcase_transfer_3 = {
    "test_name": "FOOTER + Transfer (no email address in meta)",
    "from": "email_sender@test.fr",
    "body": "Envoyé de mon iphone\nDe:\nJohn Doe\nDate:\n3 mars 2023 à 16:42:50\nBonjour\nbla bla",
    "transferred_email_processor_expected": {
        "messages.text": ["Bonjour\nbla bla"],
        "messages.meta": ["De:\nJohn Doe\nDate:\n3 mars 2023 à 16:42:50"],
        "det_original_from": None,
    },
}

testcase_transfer_4 = {
    "test_name": "BODY + Transfer",
    "from": "email_sender@test.fr",
    "body": "Ceci est un BODY\nDe:\nsender.before.transfer@test.fr\nDate:\n3 mars 2023 à 16:42:50\nBonjour\nbla bla",
    "transferred_email_processor_expected": {
        "messages.text": ["Ceci est un BODY", "Bonjour\nbla bla"],
        "messages.meta": ["", "De:\nsender.before.transfer@test.fr\nDate:\n3 mars 2023 à 16:42:50"],
        "det_original_from": None,
    },
}

testcase_false_thanks = dict(
    test_name="Thanks with question mark",
    body="Bonjour\nQu'en est-il svp ? Merci\nJoe Dupont",
    thanks_detector_expected={
        "thanks_result": False,
    },
)

testcase_basic_thanks = dict(
    test_name="Basic Thanks",
    body="Bonjour\nMerci pour cette réponse\nSincèrement\nBLA Bla BLA",
    thanks_detector_expected={
        "thanks_result": True,
    },
)


testcase_false_vacation_reply = dict(
    test_name="Simple vacation reply (False)",
    body="Bonjour\nQu'en est-il svp ? Merci\nJoe Dupont",
    vacation_reply_detector_expected={
        "vacation_reply_result": False,
    },
)

testcase_true_vacation_reply = dict(
    test_name="Simple vacation reply (True)",
    body="Bonjour, \nActuellement en congé je prendrai connaissance"
    + " de votre message ultérieurement.\nCordialement,",
    vacation_reply_detector_expected={
        "vacation_reply_result": True,
    },
)


testcase_real_message = dict(
    test_name="real_email_1",
    body=(
        "De :\n_Délégation - Ville <ma-ville@maif.fr>\n\n\nEnvoyé :\nlundi 27 septembre 2021 22:19\n\n\n"
        "À :\nTEST <test@maif.fr>\n\n\nObjet :\nTR : Soc : 1111111A - Votre"
        " attestation assurance\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nDe :\nJean Dupont <jdupont@gmail.com>"
        "\n\n\nEnvoyé :\nlundi 27 septembre 2021 22:23:43 (UTC+01:00) Brussels, Copenhagen, Madrid, Paris"
        "\n\n\nÀ :\n_Délégation - Ville <ma-ville06@maif.fr>\n\n\nSujet :\nRE: Soc : 1111111A - "
        "Votre attestation assurance\n\n\n\n\n\n\n\n\n\n\n\n\n\nBonjour,\n\n\n\n\nJe vous remercie"
        "pour votre attestation.\n\n\n\n\nLe notaire chante\n\n\n\n\n\n\n\n\n\n\nCordialement,\n\n\n\n\n"
        "Jean Dupont\n\n\n\n\n\n\n\n\n\n\n\nDe :\nma-ville@maif.fr <ma-ville@maif.fr>\n\n\n"
        "Envoyé :\nvendredi 24 septembre 2021 17:42\n\n\nÀ :\njdupont@gmail.com\n\n\nObjet :\n"
        "Soc : 1111111A - Votre attestation assurance\n\n\n\n\n\n\n\n\n\nBonjour,\n\n"
        "Veuillez trouver ci-joint l'attestation.\n\nLa visualisation des fichiers PDF nécessite "
        "Adobe Reader.\n\nSentiments mutualistes.\n\nLa MAIF"
    ),
    messages=[
        "",
        (
            "Bonjour,\n\n\n\n\nJe vous remercie pour votre attestation.\n\n\n\n\n"
            "Le notaire chante\n\n\n\n\n\n\n\n\n\n\nCordialement,\n\n\n\n\nJean Dupont"
        ),
        (
            "Bonjour,\n\nVeuillez trouver ci-joint l'attestation.\n\nLa visualisation des fichiers PDF "
            "nécessite Adobe Reader.\n\nSentiments mutualistes.\n\nLa MAIF"
        ),
    ],
)


testcase_true_reply = dict(
    test_name="Replydetecteur (True)",
    header="Re: Suivi de dossier",
    reply_detector_expected={
        "reply_result": True,
    },
)

testcase_true_reply1 = dict(
    test_name="Replydetecteur (True)",
    header="re: Suivi de dossier",
    reply_detector_expected={
        "reply_result": True,
    },
)
testcase_false_reply = dict(
    test_name="Replydetecteur (false)",
    header="tr: Suivi de dossier",
    reply_detector_expected={
        "reply_result": False,
    },
)
testcase_false_reply1 = dict(
    test_name="Replydetecteur (false)",
    header="",
    reply_detector_expected={
        "reply_result": False,
    },
)
testcase_true_transfer = dict(
    test_name="Transferdetecteur (True)",
    header="Tr: Suivi de dossier",
    body="Bonjour,\n\n\n\n\n\nUn taux d’humidité de 30% a été relevé le 01/01/2024.\n\n\n\nNous reprendrons contact avec l’assurée"
    + " en Janvier 2024.\n\n\n\n\n\n\nBien cordialement,\n\n\n\n\n\nNuméro Auxiliaire : 000000A / 000000B\n\n\n\n\n\n\n\n\n"
    + "Smith KIM\n\n\nTEST\n-\n\nL\na\nV\nalorisation du\nP\natrimoine\n\n\n2, rue du Test\n\n\n00000 Niort"
    + "\n\n\n\n\n\n\n\nTél :    0101010101\n\n\nPort :  0101010101\n\n\nhttp://test.fr\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
    + "\n\n\nDe :\nAccueil - Alex Dupond <alex@test.fr>\n\n\n\nEnvoyé :\njeudi 01 janvier 2024 01:01\n\n\nÀ "
    + ":\nCommercial <etudes@test.fr>\n\n\nObjet :\nTR: Evt : A000000000B survenu le 15/10/2021 - Intervention entreprise"
    + " partenaire\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nDe :\n\n\ntest@maif.fr\n[\nmailto:test@maif.fr\n]\n\n\n\n"
    + "Envoyé :\njeudi 01 janvier 2024 01:01\n\n\nÀ :\nAccueil - Alex Dupond\n\n\nObjet :\nEvt : A000000000B survenu le 01/01/2024"
    + " - Intervention entreprise partenaire\n\n\n\n\n\nMerci de bien vouloir prendre connaissance du document ci-joint.",
    transfer_detector_expected={
        "transfer_result": True,
    },
)
testcase_true_transfer1 = dict(
    test_name="Transferdetecteur (True)",
    header="Suivi de dossier",
    body="De :\nAccueil - Alex Dupond <accueil@test.fr>\n\n\n\nEnvoyé :\njeudi 01 janvier 2024 01:01\n\n\nÀ "
    + ":\nCommercial <etudes@test.fr>\n\n\nObjet :\nTR: Evt : A000000000B survenu le 15/10/2021 - Intervention entreprise"
    + " partenaire\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nDe :\n\n\ntest@maif.fr\n[\nmailto:test@maif.fr\n]\n\n\n\n"
    + "Envoyé :\njeudi 01 janvier 2024 01:01\n\n\nÀ :\nAccueil - Alex Dupond\n\n\nObjet :\nEvt : A000000000B survenu le 01/01/2024"
    + " - Intervention entreprise partenaire\n\n\n\n\n\nMerci de bien vouloir prendre connaissance du document ci-joint.",
    transfer_detector_expected={
        "transfer_result": True,
    },
)
testcase_false_transfer = dict(
    test_name="Transferdetecteur (False)",
    header="test",
    body="Bonjour, ceci est un message de test",
    transfer_detector_expected={
        "transfer_result": False,
    },
)
testcase_false_transfer1 = dict(
    test_name="Transferdetecteur (False)",
    header="",
    body="",
    transfer_detector_expected={
        "transfer_result": False,
    },
)

testcase_list = [value for key, value in locals().items() if key.startswith("testcase")]


def get_fixture_name(fixture_value):
    return fixture_value.get("test_name", "missing_test_name")


@pytest.fixture(
    params=testcase_list,
    ids=get_fixture_name,
)
def testcase(request, default_pipeline="my_pipeline"):
    testcase = request.param

    # Set default testcase parameters
    testcase["pipeline"] = testcase.get("pipeline", default_pipeline)

    # Set default email fields
    testcase["body"] = testcase.get("body", "")
    testcase["header"] = testcase.get("header", "")
    testcase["from"] = testcase.get("from", "")
    testcase["to"] = testcase.get("to", "")
    testcase["attachments"] = testcase.get("attachments", list())

    return testcase
