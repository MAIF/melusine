import re

import pytest

from melusine.message import Message
from melusine.processors import BaseContentTagger, ContentTagger, Tag


def test_content_tagger():
    # Text segments (= individual messages in an email conversation)
    text_segments = [
        "Envoye de mon iphone",
        ("Bonjour Mme X,\nSuite a blh blah blah\n" "Bien cordialement\nJane Dupond\n" "(See attached file: flex.jpg)"),
        (
            "Bonjour,\nVeuillez trouver ci-joint blah\n"
            "Merci d'avance,\nCordialement,\n"
            "Toute modification, edition, utilisation ou diffusion non autorisee est interdite"
        ),
    ]

    # Expected tags
    expected_tags = [
        [
            ("FOOTER", "Envoye de mon iphone"),
        ],
        [
            ("HELLO", "Bonjour Mme X,"),
            ("BODY", "Suite a blh blah blah"),
            ("GREETINGS", "Bien cordialement"),
            ("SIGNATURE_NAME", "Jane Dupond"),
            ("PJ", "(See attached file: flex.jpg)"),
        ],
        [
            ("HELLO", "Bonjour,"),
            ("BODY", "Veuillez trouver ci-joint blah"),
            ("THANKS", "Merci d'avance,"),
            ("GREETINGS", "Cordialement,"),
            (
                "FOOTER",
                "Toute modification, edition, utilisation ou diffusion non autorisee est interdite",
            ),
        ],
    ]

    # Mock the output of a Segmenter (List of Message object)
    messages = [Message(text=segment) for segment in text_segments]

    # Instantiate and apply the Tagger
    tagger = ContentTagger()
    output_messages = tagger.tag_email(messages)

    # Test output tags
    output_tags = [x.tags for x in output_messages]
    assert output_tags == expected_tags


def test_tag_null_message():
    messages = None

    # Instantiate and apply the Tagger
    tagger = ContentTagger()
    output_messages = tagger.tag_email(messages)

    assert output_messages is None


@pytest.mark.parametrize(
    "text, expected_parts",
    [
        (
            "Bonjour, merci pour votre message!\nComment-allez vous?! Je suis satisfait!!!\n"
            "Bien cordialement\n\n\n\nJane Dupond\n",
            [
                "Bonjour,",
                "merci pour votre message!",
                "Comment-allez vous?!",
                "Je suis satisfait!!!",
                "Bien cordialement",
                "Jane Dupond",
            ],
        ),
    ],
)
def test_content_tagger_split_text(text, expected_parts):
    # Instantiate and apply the Tagger
    tagger = ContentTagger()
    output_parts = tagger.split_text(text)

    assert output_parts == expected_parts


@pytest.mark.parametrize(
    "text, expected_tags",
    [
        (
            "Bonjour Mme X,\nSuite a blh blah blah.\n"
            "Bien cordialement\nJane Dupond\n"
            "(See attached file: flex.jpg)",
            [
                ("HELLO", "Bonjour Mme X,"),
                ("BODY", "Suite a blh blah blah."),
                ("GREETINGS", "Bien cordialement"),
                ("SIGNATURE_NAME", "Jane Dupond"),
                ("PJ", "(See attached file: flex.jpg)"),
            ],
        ),
        (
            "Bonjour, je confirme le rdv. Cordialement, John Smith",
            [
                ("HELLO", "Bonjour,"),
                ("BODY", "je confirme le rdv."),
                ("GREETINGS", "Cordialement, John Smith"),
            ],
        ),
        (
            (
                "Bonjour,\nSuite a notre intervention du 16.02.22 , un taux d'humidité de 50% a été relevé.\n"
                "Cordialement.\n177, rue de la fée - 75000 Paris.\n"
                "Horaires : du lundi au jeudi de 08h00 à 16h30 et le vendredi de 08h00 à 16h00.\n"
                "Tel : 01.45.53.11.33"
            ),
            [
                ("HELLO", "Bonjour,"),
                ("BODY", "Suite a notre intervention du 16.02.22 , un taux d'humidité de 50% a été relevé."),
                ("GREETINGS", "Cordialement."),
                ("SIGNATURE", "177, rue de la fée - 75000 Paris."),
                ("BODY", "Horaires : du lundi au jeudi de 08h00 à 16h30 et le vendredi de 08h00 à 16h00."),
                ("SIGNATURE", "Tel : 01.45.53.11.33"),
            ],
        ),
        (
            (
                "bonjour\n"
                "15 jours après les premières réparations, un défaut a été détecté. "
                "Bien à vous\n"
                "Britney Spears"
            ),
            [
                ("HELLO", "bonjour"),
                ("BODY", "15 jours après les premières réparations, un défaut a été détecté."),
                ("GREETINGS", "Bien à vous"),
                ("SIGNATURE_NAME", "Britney Spears"),
            ],
        ),
        (
            (
                "Bonjour monsieur Smith\n"
                "merci. Bien à vous\n"
                "Britney Spears\n"
                "22 hollywood boulevard\n"
                "79000 Niort\n"
            ),
            [
                ("HELLO", "Bonjour monsieur Smith"),
                ("THANKS", "merci."),
                ("GREETINGS", "Bien à vous"),
                ("SIGNATURE_NAME", "Britney Spears"),
                ("SIGNATURE", "22 hollywood boulevard"),
                ("SIGNATURE", "79000 Niort"),
            ],
        ),
        (
            (
                "Merci de me faire suivre les docs à ma nouvelle adresse qui est 0 rue du parc, 75000 Paris. "
                "Merci d'avance. \nAcceptez notre salutation,"
            ),
            [
                ("BODY", "Merci de me faire suivre les docs à ma nouvelle adresse qui est 0 rue du parc, 75000 Paris."),
                ("THANKS", "Merci d'avance."),
                ("GREETINGS", "Acceptez notre salutation,"),
            ],
        ),
        (
            (
                "Bonjour\n"
                "Je vous relance concernant ma télévision avec le devis en PJ.\n"
                "Désolé pour la qualité.\n"
                "Je l'ai envoyé à partir de mon ordi.\n"
                "Excellente journée à vous,\n"
                "Bon we\n"
                "Votre bien dévoué\n"
                "amicalement votre\n"
                "Cordiales salutations.\n"
                "Françoise-Bénédicte Dupond\n"
                "Envoyé à partir de \nCourrier \npour Windows"
            ),
            [
                ("HELLO", "Bonjour"),
                ("BODY", "Je vous relance concernant ma télévision avec le devis en PJ."),
                ("BODY", "Désolé pour la qualité."),
                ("BODY", "Je l'ai envoyé à partir de mon ordi."),
                ("HELLO", "Excellente journée à vous,"),
                ("HELLO", "Bon we"),
                ("GREETINGS", "Votre bien dévoué"),
                ("GREETINGS", "amicalement votre"),
                ("GREETINGS", "Cordiales salutations."),
                ("SIGNATURE_NAME", "Françoise-Bénédicte Dupond"),
                ("FOOTER", "Envoyé à partir de"),
                ("FOOTER", "Courrier"),
                ("FOOTER", "pour Windows"),
            ],
        ),
        (
            "C'est bien note, merci beaucoup.\nSentiments dévoués.\nTélécharger \nOutlook pour Android",
            [
                ("THANKS", "C'est bien note, merci beaucoup."),
                ("GREETINGS", "Sentiments dévoués."),
                ("FOOTER", "Télécharger"),
                ("FOOTER", "Outlook pour Android"),
            ],
        ),
        (
            "Impeccable, je vous remercie beaucoup pour votre rapidité.\nObtenir\nOutlook pour Android",
            [
                ("THANKS", "Impeccable, je vous remercie beaucoup pour votre rapidité."),
                ("FOOTER", "Obtenir"),
                ("FOOTER", "Outlook pour Android"),
            ],
        ),
        (
            (
                "Cher Monsieur,\nJe vous confirme la bonne réception de votre précédent email.\n"
                "Je vous en remercie.\nBien cordialement,\nJohn Smith"
            ),
            [
                ("HELLO", "Cher Monsieur,"),
                ("BODY", "Je vous confirme la bonne réception de votre précédent email."),
                ("THANKS", "Je vous en remercie."),
                ("GREETINGS", "Bien cordialement,"),
                ("SIGNATURE_NAME", "John Smith"),
            ],
        ),
        (
            (
                "chère madame,\n"
                "URGENT URGENT\n"
                "Merci de me faire suivre les docs à ma nouvelle adresse qui est 0 rue du parc, 75000 Paris. "
                "Merci d'avance. \nRecevez nos salutations,\nVous en souhaitant bonne réception"
            ),
            [
                ("HELLO", "chère madame,"),
                ("BODY", "URGENT URGENT"),
                ("BODY", "Merci de me faire suivre les docs à ma nouvelle adresse qui est 0 rue du parc, 75000 Paris."),
                ("THANKS", "Merci d'avance."),
                ("GREETINGS", "Recevez nos salutations,"),
                ("GREETINGS", "Vous en souhaitant bonne réception"),
            ],
        ),
        pytest.param(
            "Un témoignage sous X\nEnvoyé depuis mon téléphone Orange",
            [
                ("BODY", "Un témoignage sous X"),
                ("FOOTER", "Envoyé depuis mon téléphone Orange"),
            ],
            id="Edge case where a line ends with an isolated character",
        ),
        pytest.param(
            "     ??\n  !??!",
            [
                ("BODY", "??!??!"),
            ],
            id="Edge case where the two first lines are missing word characters",
        ),
        (
            "Bonjour Mme X,\nSuite a blh blah blah.\n"
            "Bien cordialement\nJane Dupond\n"
            "(See attached file: flex.jpg)",
            [
                ("HELLO", "Bonjour Mme X,"),
                ("BODY", "Suite a blh blah blah."),
                ("GREETINGS", "Bien cordialement"),
                ("SIGNATURE_NAME", "Jane Dupond"),
                ("PJ", "(See attached file: flex.jpg)"),
            ],
        ),
        (
            "\nChère Madame\n\nC'est bien noté, merci\nBien reçu\nJ.Smith\n\n",
            [
                ("HELLO", "Chère Madame"),
                ("THANKS", "C'est bien noté, merci"),
                ("BODY", "Bien reçu"),
                ("SIGNATURE_NAME", "J.Smith"),
            ],
        ),
        (
            "\nBonjour Monsieur, ceci n'est pas un hello\nBonne fin de journee\nsalutations",
            [
                ("BODY", "Bonjour Monsieur, ceci n'est pas un hello"),
                ("HELLO", "Bonne fin de journee"),
                ("GREETINGS", "salutations"),
            ],
        ),
        (
            "\nBonjour Monsieur Stanislas von den hoeggenboord\n\nbien à toi\nJ.  Smith\nChargé de clientèle",
            [
                ("HELLO", "Bonjour Monsieur Stanislas von den hoeggenboord"),
                ("GREETINGS", "bien à toi"),
                ("SIGNATURE_NAME", "J. Smith"),
                ("SIGNATURE", "Chargé de clientèle"),
            ],
        ),
        (
            (
                "\n1 rdv à 18h\n\n2 ème message laissé à la locataire\n3je m'en vais au bois\n"
                "4 allée des iris\n 5bis rue Patrick Sebastien\n6-8 cours mirabeau\n 7 ter place du dahu\n"
                "8 de la rue très longue qui ne doit pas être taggée signature"
            ),
            [
                ("BODY", "1 rdv à 18h"),
                ("BODY", "2 ème message laissé à la locataire"),
                ("BODY", "3je m'en vais au bois"),
                ("SIGNATURE", "4 allée des iris"),
                ("SIGNATURE", "5bis rue Patrick Sebastien"),
                ("SIGNATURE", "6-8 cours mirabeau"),
                ("SIGNATURE", "7 ter place du dahu"),
                ("BODY", "8 de la rue très longue qui ne doit pas être taggée signature"),
            ],
        ),
        (
            (
                "à L'attention de M Bob,\n"
                "Bonjour,\n"
                "Je vous informe que je vais accepter la proposition de L , à savoir le paiement d'une indemnité forfaitaire de résiliation  du CCMI de  4000 € TTC pour clore cette affaire.\n"
                "Cordialement.\n"
                "Bob Smith"
            ),
            [
                ("FOOTER", "à L'attention de M Bob,"),
                ("HELLO", "Bonjour,"),
                (
                    "BODY",
                    "Je vous informe que je vais accepter la proposition de L , à savoir le paiement d'une indemnité forfaitaire de résiliation du CCMI de 4000 € TTC pour clore cette affaire.",
                ),
                ("GREETINGS", "Cordialement."),
                ("SIGNATURE_NAME", "Bob Smith"),
            ],
        ),
        (
            (
                "Monsieur Bob Smith\n"
                "Adresse mail : BobSmith90@gmail.com\n"
                "Lucy Ange\n\n"
                "Bonjour Monsieur,\n"
                "Suite à notre entretien téléphonique de ce matin, et au message que vous m'avez envoyé sur ma messagerie, je voudrais effectuer la réparation du véhicule Renault Twingo dans un garage partenaire de la Maif situé, si c'est possible.\n"
                "Dans l'attente de votre réponse et en vous remerciant par avance,\n\n\n"
                "Monsieur Bob Smith\n\n\n"
                "Envoyé à partir de\n"
                "Courrier\npour Windows\n\n\n\n"
                "Sans virus.\nwww.avast.com"
            ),
            [
                ("HELLO", "Monsieur Bob Smith"),
                ("SIGNATURE", "Adresse mail : BobSmith90@gmail.com"),
                ("SIGNATURE_NAME", "Lucy Ange"),
                ("HELLO", "Bonjour Monsieur,"),
                (
                    "BODY",
                    "Suite à notre entretien téléphonique de ce matin, et au message que vous m'avez envoyé sur ma messagerie, je voudrais effectuer la réparation du véhicule Renault Twingo dans un garage partenaire de la Maif situé, si c'est possible.",
                ),
                ("BODY", "Dans l'attente de votre réponse et en vous remerciant par avance,"),
                ("HELLO", "Monsieur Bob Smith"),
                ("FOOTER", "Envoyé à partir de"),
                ("FOOTER", "Courrier"),
                ("FOOTER", "pour Windows"),
                ("FOOTER", "Sans virus."),
                ("FOOTER", "www.avast.com"),
            ],
        ),
        (
            (
                "Bob Smith\n\n\n"
                "A l’attention de Madame Lucy Ange,\n\n\n\n\n\n"
                "Bonjour Madame Ange,\n\n\n\n\n\n\n\n\n"
                "J’espère que vous allez bien.\n\n\n\n\n\n"
                "Pour faire suite à mon mail du 21 février 2023, je me permets de revenir vers vous pour avoir votre avis sur le devis que j’ai demandé auprès d’un enquêteur.\n\n\n\n"
                "Voici son retour :\n\n\n\n\n\n"
                "Qu’en pensez-vous svp ?\n\n\n\n\n\n"
                "Je reste à votre disposition pour tout complément d’information et vous remercie de l’intérêt que vous porterez à ma demande,\n\n\n\n\n\n"
                "Bien Cordialement,\n\n\n\n\n\n"
                "Bob Smith\n\n\n"
                "Tél. 06.83.22.95.94"
            ),
            [
                ("SIGNATURE_NAME", "Bob Smith"),
                ("FOOTER", "A l’attention de Madame Lucy Ange,"),
                ("HELLO", "Bonjour Madame Ange,"),
                ("BODY", "J’espère que vous allez bien."),
                (
                    "BODY",
                    "Pour faire suite à mon mail du 21 février 2023, je me permets de revenir vers vous pour avoir votre avis sur le devis que j’ai demandé auprès d’un enquêteur.",
                ),
                ("BODY", "Voici son retour :"),
                ("BODY", "Qu’en pensez-vous svp ?"),
                (
                    "BODY",
                    "Je reste à votre disposition pour tout complément d’information et vous remercie de l’intérêt que vous porterez à ma demande,",
                ),
                ("GREETINGS", "Bien Cordialement,"),
                ("SIGNATURE_NAME", "Bob Smith"),
                ("SIGNATURE", "Tél."),
                ("SIGNATURE", "06.83.22.95.94"),
            ],
        ),
        pytest.param(
            (
                "cordialement\nContact e-mail\n\n\nContact téléphone\n\n01 23 45 67 89 / abcabc@hotmail.fr\n"
                "Torroella de Montgri, le 5 avril 2023\nLes formats de fichiers acceptés sont : PDF, DOC, DOCX, JPEG, "
                "JPG, TIFF, TXT, ODT, XLS, XLSX\nTout autre format de fichiers ne sera pas transmis au dossier"
            ),
            [
                ("GREETINGS", "cordialement"),
                ("SIGNATURE", "Contact e-mail"),
                ("SIGNATURE", "Contact téléphone"),
                ("SIGNATURE", "01 23 45 67 89 / abcabc@hotmail.fr"),
                ("SIGNATURE", "Torroella de Montgri, le 5 avril 2023"),
                (
                    "FOOTER",
                    "Les formats de fichiers acceptés sont : PDF, DOC, DOCX, JPEG, JPG, TIFF, TXT, ODT, XLS, XLSX",
                ),
                ("FOOTER", "Tout autre format de fichiers ne sera pas transmis au dossier"),
            ],
            id="diverse_signature_patterns",
        ),
        pytest.param(
            (
                "bonjour\nmon body\nJ. Smith\n\n01 23 45 67 89\nSecrétaire en charge des avions\n"
                "Business Analyst – Tribu Sinistres – Squad Flux Entrants\n"
                "Société nationale des chemins de fer\nConseiller MAIF\nGestionnaire sinistre - C99G\n"
                "Service des lettres anonymes\nTechnicienne de gestion - EQUIPE ABC\n"
            ),
            [
                ("HELLO", "bonjour"),
                ("BODY", "mon body"),
                ("SIGNATURE_NAME", "J. Smith"),
                ("SIGNATURE", "01 23 45 67 89"),
                ("SIGNATURE", "Secrétaire en charge des avions"),
                ("SIGNATURE", "Business Analyst – Tribu Sinistres – Squad Flux Entrants"),
                ("SIGNATURE", "Société nationale des chemins de fer"),
                ("SIGNATURE", "Conseiller MAIF"),
                ("SIGNATURE", "Gestionnaire sinistre - C99G"),
                ("SIGNATURE", "Service des lettres anonymes"),
                ("SIGNATURE", "Technicienne de gestion - EQUIPE ABC"),
            ],
            id="signature_jobs",
        ),
        pytest.param(
            (
                "bonjour\nmon body\nCordialement\n\n"
                "analyste -------------------------------------- test test test test test test test\n"
            ),
            [
                ("HELLO", "bonjour"),
                ("BODY", "mon body"),
                ("GREETINGS", "Cordialement"),
                ("BODY", "analyste -------------------------------------- test test test test test test test"),
            ],
            id="check_catastrophic_backtracking",
        ),
    ],
)
def test_tag_text_generic(text, expected_tags):
    # Instantiate and apply the Tagger
    tagger = ContentTagger()
    output_tags = tagger.tag_text(text)
    # Test output tags
    assert output_tags == expected_tags


@pytest.mark.parametrize(
    "text, expected_tags",
    [
        pytest.param(
            (
                "Merci\n"
                "Je vous remercie\n"
                "Merci d'avance\n"
                "Je vous remercie par avance\n"
                "Vous en remerciant par avance.\n"
            ),
            [
                ("THANKS", "Merci"),
                ("THANKS", "Je vous remercie"),
                ("THANKS", "Merci d'avance"),
                ("THANKS", "Je vous remercie par avance"),
                ("THANKS", "Vous en remerciant par avance."),
            ],
            id="french thanks patterns",
        ),
    ],
)
def test_tag_text_french(text, expected_tags):
    # Instantiate and apply the Tagger
    tagger = ContentTagger()
    output_tags = tagger.tag_text(text)
    # Test output tags
    assert output_tags == expected_tags


@pytest.mark.parametrize(
    "text, expected_tags",
    [
        pytest.param(
            (
                "Thank you so much\n"
                "thanks\n"
                "thx Joanna\n"
                "thanks but you forgot bla\n"
                "Thx however I still need the document\n"
            ),
            [
                ("THANKS", "Thank you so much"),
                ("THANKS", "thanks"),
                ("THANKS", "thx Joanna"),
                ("BODY", "thanks but you forgot bla"),
                ("BODY", "Thx however I still need the document"),
            ],
            id="english thanks patterns",
        ),
        pytest.param(
            (
                "Best\n"
                "warm Wishes\n"
                "regards\n"
                "best regards\n"
                "cheers\n"
                "yours\n"
                "yours truly\n"
                "Sincerely\n"
                "see you soon\n"
                "Speak to you soon\n"
                "talk soon\n"
                "Take care\n"
                "Catch you later\n"
                "Have a fantastic day\n"
                "Looking forward to your reply\n"
                "I am looking forward to hearing from you\n"
                "Hoping to hear from you\n"
            ),
            [
                ("GREETINGS", "Best"),
                ("GREETINGS", "warm Wishes"),
                ("GREETINGS", "regards"),
                ("GREETINGS", "best regards"),
                ("GREETINGS", "cheers"),
                ("GREETINGS", "yours"),
                ("GREETINGS", "yours truly"),
                ("GREETINGS", "Sincerely"),
                ("GREETINGS", "see you soon"),
                ("GREETINGS", "Speak to you soon"),
                ("GREETINGS", "talk soon"),
                ("GREETINGS", "Take care"),
                ("GREETINGS", "Catch you later"),
                ("GREETINGS", "Have a fantastic day"),
                ("GREETINGS", "Looking forward to your reply"),
                ("GREETINGS", "I am looking forward to hearing from you"),
                ("GREETINGS", "Hoping to hear from you"),
            ],
            id="english greetings",
        ),
        pytest.param(
            (
                "Hello John\n"
                "hi\n"
                "Hi there\n"
                "good to hear from you\n"
                "it is good to hear from you\n"
                "I hope you are having a great week\n"
                "how are you doing\n"
                "how are you positioned about the matter\n"
                "i hope you are doing well\n"
                "Good Morning Joanna\n"
                "good Afternoon\n"
                "Dear Jacky\n"
                "Sir\n"
                "Dear Madam\n"
                "Dear Mr\n"
                "Dear Ms.\n"
                "Dear miss\n"
                "Dear mrs.\n"
                "Dear sir or madam\n"
                "To whom it may concern\n"
            ),
            [
                ("HELLO", "Hello John"),
                ("HELLO", "hi"),
                ("HELLO", "Hi there"),
                ("HELLO", "good to hear from you"),
                ("HELLO", "it is good to hear from you"),
                ("HELLO", "I hope you are having a great week"),
                ("HELLO", "how are you doing"),
                ("BODY", "how are you positioned about the matter"),
                ("HELLO", "i hope you are doing well"),
                ("HELLO", "Good Morning Joanna"),
                ("HELLO", "good Afternoon"),
                ("HELLO", "Dear Jacky"),
                ("HELLO", "Sir"),
                ("HELLO", "Dear Madam"),
                ("HELLO", "Dear Mr"),
                ("HELLO", "Dear Ms."),
                ("HELLO", "Dear miss"),
                ("HELLO", "Dear mrs."),
                ("HELLO", "Dear sir or madam"),
                ("HELLO", "To whom it may concern"),
            ],
            id="english hello",
        ),
        pytest.param(
            (
                "VP of Data Science\n"
                "Chief of staff\n"
                "CTO at TestMelusine\n"
                "CEOABC test\n"
                "Lead business developer\n"
            ),
            [
                ("SIGNATURE", "VP of Data Science"),
                ("SIGNATURE", "Chief of staff"),
                ("SIGNATURE", "CTO at TestMelusine"),
                ("BODY", "CEOABC test"),
                ("SIGNATURE", "Lead business developer"),
            ],
            id="english job signature patterns",
        ),
        pytest.param(
            (
                "9 downing street\n"
                "4-6 Beverly Hill\n"
                "4 Abbey road W24RA\n"
                "3 Ocean Rd.\n"
                "5th avenue\n"
                "221b Baker St.\n"
                "6bis River ln.\n"
                "7 Winter lane\n"
            ),
            [
                ("SIGNATURE", "9 downing street"),
                ("SIGNATURE", "4-6 Beverly Hill"),
                ("SIGNATURE", "4 Abbey road W24RA"),
                ("SIGNATURE", "3 Ocean Rd."),
                ("SIGNATURE", "5th avenue"),
                ("SIGNATURE", "221b Baker St."),
                ("SIGNATURE", "6bis River ln."),
                ("SIGNATURE", "7 Winter lane"),
            ],
            id="english adsress signature patterns",
        ),
    ],
)
def test_tag_text_english(text, expected_tags):
    # Instantiate and apply the Tagger
    tagger = ContentTagger()
    output_tags = tagger.tag_text(text)
    # Test output tags
    assert output_tags == expected_tags


def test_tag_list():
    # Limit tags to "HELLO" and the default tag ("BODY")
    tag_list = ["HELLO"]

    # Text segment (= individual message in an email conversation)
    text = "bonjour\nblah blah blah\nmerci\ncordialement"

    # Expected tags
    expected_tags = [
        ("HELLO", "bonjour"),
        ("BODY", "blah blah blah"),
        ("BODY", "merci"),
        ("BODY", "cordialement"),
    ]

    # Instantiate and apply the Tagger
    tagger = ContentTagger(tag_list=tag_list)
    output_tags = tagger.tag_text(text)

    # Test output tags
    assert expected_tags == output_tags


def test_undefined_tag():
    unknown_tag = "UNKNOWN_TAG"

    # Setup an unknown tag
    tag_list = [unknown_tag]

    # Instantiate Tagger
    with pytest.raises(ValueError, match=rf".*{unknown_tag}.*"):
        _ = ContentTagger(tag_list=tag_list)


def test_unsupported_type():
    class MyClass(ContentTagger):
        """Test class"""

        @Tag
        def TEST_TAG(self):
            """Test method"""
            return 3.3

    with pytest.raises(ValueError, match="supported types"):
        _ = MyClass()


def test_compiled_pattern():
    class MyClass(ContentTagger):
        """Test class"""

        @Tag
        def TEST_TAG(self):
            """Test method"""
            return re.compile(r"cool_pattern")

    tagger = MyClass()
    subtext, tag, match = tagger("cool_pattern is what I am looking for")[0]

    # Check tag result
    assert tag == "TEST_TAG"


def test_str_pattern():
    class MyClass(ContentTagger):
        """Test class"""

        @Tag
        def TEST_TAG(self):
            """Test method"""
            return r"cool_pattern"

    tagger = MyClass()
    subtext, tag, match = tagger("cool_pattern is what I am looking for")[0]

    # Check tag result
    assert tag == "TEST_TAG"


def test_malformed_regex():
    from melusine.processors import Tag

    malformed_regex = r"[*."

    # Create a tagger containing an ill defined Tag (malformed regex)
    class CustomTagger(ContentTagger):
        """Test class"""

        @Tag
        def HELLO(self):
            """Test method"""
            return malformed_regex

    # Instantiate Tagger
    with pytest.raises(ValueError, match=rf"Invalid regex"):
        _ = CustomTagger()


def test_direct_tagging():
    tagger = ContentTagger()
    match = tagger["HELLO"].match("Bonjour")

    assert bool(match)


def test_call_method():
    tagger = ContentTagger()

    match_list = tagger("Bonjour a tous")
    subtext, tag, regex = match_list[0]

    assert tag == "HELLO"


@pytest.mark.parametrize(
    "text, n_words, word_character_only, expected_match",
    [
        pytest.param("Hello you", 4, False, True, id="4 words match"),
        pytest.param("Hello how are you today", 4, False, False, id="4 words no match"),
        pytest.param("Hello! you?", 4, False, True, id="4 words match with special characters"),
        pytest.param(
            "Hello! you?", 4, True, False, id="4 words match with special characters (word character only True)"
        ),
    ],
)
def test_word_blocks(text, n_words, word_character_only, expected_match):
    regex = BaseContentTagger.word_block(n_words, word_character_only=word_character_only)

    search_regex = r"^" + regex + r"$"
    match = bool(re.search(search_regex, text))
    assert match == expected_match
