import re

import pytest

from melusine.message import Message
from melusine.processors import BaseContentTagger, ContentTagger, RefinedTagger, Tag


def test_content_tagger():
    # Text segments (= individual messages in an email conversation)
    text_segments = [
        "Envoye de mon iphone",
        "Bonjour Mme X,\nSuite a blh blah blah\nBien cordialement\nJane Dupond\n(See attached file: flex.jpg)",
        (
            "Bonjour,\nVeuillez trouver ci-joint blah\n"
            "Merci d'avance,\nCordialement,\n"
            "Toute modification, edition, utilisation ou diffusion non autorisee est interdite"
        ),
    ]

    # Expected tags
    expected_tags = [
        [
            {"base_text": "Envoye de mon iphone", "base_tag": "FOOTER"},
        ],
        [
            {"base_text": "Bonjour Mme X,", "base_tag": "HELLO"},
            {"base_text": "Suite a blh blah blah", "base_tag": "BODY"},
            {"base_text": "Bien cordialement", "base_tag": "GREETINGS"},
            {"base_text": "Jane Dupond", "base_tag": "BODY"},
            {"base_text": "(See attached file: flex.jpg)", "base_tag": "PJ"},
        ],
        [
            {"base_text": "Bonjour,", "base_tag": "HELLO"},
            {"base_text": "Veuillez trouver ci-joint blah", "base_tag": "BODY"},
            {"base_text": "Merci d'avance,", "base_tag": "THANKS"},
            {"base_text": "Cordialement,", "base_tag": "GREETINGS"},
            {
                "base_text": "Toute modification, edition, utilisation ou diffusion non autorisee est interdite",
                "base_tag": "FOOTER",
            },
        ],
    ]

    # Mock the output of a Segmenter (List of Message object)
    messages = [Message(text=segment) for segment in text_segments]

    # Instantiate and apply the Tagger
    tagger = ContentTagger()
    output_messages = tagger.tag_email(messages)

    # Test output tags
    for tag_data_list in expected_tags:
        for tag_data in tag_data_list:
            if "base_tag_list" not in tag_data_list:
                tag_data["base_tag_list"] = [tag_data["base_tag"]]

    for i, message in enumerate(output_messages):
        for j, tag_data in enumerate(message.tags):
            assert tag_data == expected_tags[i][j]


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
                {"base_text": "Bonjour Mme X,", "base_tag": "HELLO", "refined_tag": "HELLO"},
                {"base_text": "Suite a blh blah blah.", "base_tag": "BODY", "refined_tag": "BODY"},
                {"base_text": "Bien cordialement", "base_tag": "GREETINGS", "refined_tag": "GREETINGS"},
                {"base_text": "Jane Dupond", "base_tag": "BODY", "refined_tag": "SIGNATURE_NAME"},
                {"base_text": "(See attached file: flex.jpg)", "base_tag": "PJ", "refined_tag": "PJ"},
            ],
        ),
        (
            "Bonjour, je confirme le rdv. Cordialement, John Smith",
            [
                {"base_text": "Bonjour,", "base_tag": "HELLO", "refined_tag": "HELLO"},
                {"base_text": "je confirme le rdv.", "base_tag": "BODY", "refined_tag": "BODY"},
                {"base_text": "Cordialement, John Smith", "base_tag": "GREETINGS", "refined_tag": "GREETINGS"},
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
                {"base_text": "Bonjour,", "base_tag": "HELLO", "refined_tag": "HELLO"},
                {
                    "base_text": "Suite a notre intervention du 16.02.22 , un taux d'humidité de 50% a été relevé.",
                    "base_tag": "BODY",
                    "refined_tag": "BODY",
                },
                {"base_text": "Cordialement.", "base_tag": "GREETINGS", "refined_tag": "GREETINGS"},
                {"base_text": "177, rue de la fée - 75000 Paris.", "base_tag": "SIGNATURE", "refined_tag": "SIGNATURE"},
                {
                    "base_text": "Horaires : du lundi au jeudi de 08h00 à 16h30 et le vendredi de 08h00 à 16h00.",
                    "base_tag": "BODY",
                    "refined_tag": "BODY",
                },
                {"base_text": "Tel : 01.45.53.11.33", "base_tag": "SIGNATURE", "refined_tag": "SIGNATURE"},
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
                {"base_text": "bonjour", "base_tag": "HELLO", "refined_tag": "HELLO"},
                {
                    "base_text": "15 jours après les premières réparations, un défaut a été détecté.",
                    "base_tag": "BODY",
                    "refined_tag": "BODY",
                },
                {"base_text": "Bien à vous", "base_tag": "GREETINGS", "refined_tag": "GREETINGS"},
                {"base_text": "Britney Spears", "base_tag": "BODY", "refined_tag": "SIGNATURE_NAME"},
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
                {"base_text": "Bonjour monsieur Smith", "base_tag": "HELLO", "refined_tag": "HELLO"},
                {"base_text": "merci.", "base_tag": "THANKS", "refined_tag": "THANKS"},
                {"base_text": "Bien à vous", "base_tag": "GREETINGS", "refined_tag": "GREETINGS"},
                {"base_text": "Britney Spears", "base_tag": "BODY", "refined_tag": "SIGNATURE_NAME"},
                {"base_text": "22 hollywood boulevard", "base_tag": "SIGNATURE", "refined_tag": "SIGNATURE"},
                {"base_text": "79000 Niort", "base_tag": "SIGNATURE", "refined_tag": "SIGNATURE"},
            ],
        ),
        (
            (
                "Merci de me faire suivre les docs à ma nouvelle adresse qui est 0 rue du parc, 75000 Paris. "
                "Merci d'avance. \nAcceptez notre salutation,"
            ),
            [
                {
                    "base_text": "Merci de me faire suivre les docs à ma nouvelle adresse qui est 0 rue du parc, 75000 Paris.",
                    "base_tag": "BODY",
                    "refined_tag": "BODY",
                },
                {"base_text": "Merci d'avance.", "base_tag": "THANKS", "refined_tag": "THANKS"},
                {"base_text": "Acceptez notre salutation,", "base_tag": "GREETINGS", "refined_tag": "GREETINGS"},
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
                {"base_text": "Bonjour", "base_tag": "HELLO", "refined_tag": "HELLO"},
                {
                    "base_text": "Je vous relance concernant ma télévision avec le devis en PJ.",
                    "base_tag": "BODY",
                    "refined_tag": "BODY",
                },
                {"base_text": "Désolé pour la qualité.", "base_tag": "BODY", "refined_tag": "BODY"},
                {"base_text": "Je l'ai envoyé à partir de mon ordi.", "base_tag": "BODY", "refined_tag": "BODY"},
                {"base_text": "Excellente journée à vous,", "base_tag": "HELLO", "refined_tag": "HELLO"},
                {"base_text": "Bon we", "base_tag": "HELLO", "refined_tag": "HELLO"},
                {"base_text": "Votre bien dévoué", "base_tag": "GREETINGS", "refined_tag": "GREETINGS"},
                {"base_text": "amicalement votre", "base_tag": "GREETINGS", "refined_tag": "GREETINGS"},
                {"base_text": "Cordiales salutations.", "base_tag": "GREETINGS", "refined_tag": "GREETINGS"},
                {"base_text": "Françoise-Bénédicte Dupond", "base_tag": "BODY", "refined_tag": "SIGNATURE_NAME"},
                {"base_text": "Envoyé à partir de", "base_tag": "FOOTER", "refined_tag": "FOOTER"},
                {"base_text": "Courrier", "base_tag": "FOOTER", "refined_tag": "FOOTER"},
                {"base_text": "pour Windows", "base_tag": "FOOTER", "refined_tag": "FOOTER"},
            ],
        ),
        (
            "C'est bien note, merci beaucoup.\nSentiments dévoués.\nTélécharger \nOutlook pour Android",
            [
                {"base_text": "C'est bien note, merci beaucoup.", "base_tag": "THANKS", "refined_tag": "THANKS"},
                {"base_text": "Sentiments dévoués.", "base_tag": "GREETINGS", "refined_tag": "GREETINGS"},
                {"base_text": "Télécharger", "base_tag": "FOOTER", "refined_tag": "FOOTER"},
                {"base_text": "Outlook pour Android", "base_tag": "FOOTER", "refined_tag": "FOOTER"},
            ],
        ),
        (
            "Impeccable, je vous remercie beaucoup pour votre rapidité.\nObtenir\nOutlook pour Android",
            [
                {
                    "base_text": "Impeccable, je vous remercie beaucoup pour votre rapidité.",
                    "base_tag": "THANKS",
                    "refined_tag": "THANKS",
                },
                {"base_text": "Obtenir", "base_tag": "FOOTER", "refined_tag": "FOOTER"},
                {"base_text": "Outlook pour Android", "base_tag": "FOOTER", "refined_tag": "FOOTER"},
            ],
        ),
        (
            (
                "Cher Monsieur,\nJe vous confirme la bonne réception de votre précédent email.\n"
                "Je vous en remercie.\nBien cordialement,\nJohn Smith"
            ),
            [
                {"base_text": "Cher Monsieur,", "base_tag": "HELLO", "refined_tag": "HELLO"},
                {
                    "base_text": "Je vous confirme la bonne réception de votre précédent email.",
                    "base_tag": "BODY",
                    "refined_tag": "BODY",
                },
                {"base_text": "Je vous en remercie.", "base_tag": "THANKS", "refined_tag": "THANKS"},
                {"base_text": "Bien cordialement,", "base_tag": "GREETINGS", "refined_tag": "GREETINGS"},
                {"base_text": "John Smith", "base_tag": "BODY", "refined_tag": "SIGNATURE_NAME"},
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
                {"base_text": "chère madame,", "base_tag": "HELLO", "refined_tag": "HELLO"},
                {"base_text": "URGENT URGENT", "base_tag": "BODY", "refined_tag": "BODY"},
                {
                    "base_text": "Merci de me faire suivre les docs à ma nouvelle adresse qui est 0 rue du parc, 75000 Paris.",
                    "base_tag": "BODY",
                    "refined_tag": "BODY",
                },
                {"base_text": "Merci d'avance.", "base_tag": "THANKS", "refined_tag": "THANKS"},
                {"base_text": "Recevez nos salutations,", "base_tag": "GREETINGS", "refined_tag": "GREETINGS"},
                {
                    "base_text": "Vous en souhaitant bonne réception",
                    "base_tag": "GREETINGS",
                    "refined_tag": "GREETINGS",
                },
            ],
        ),
        pytest.param(
            "Un témoignage sous X\nEnvoyé depuis mon téléphone Orange",
            [
                {"base_text": "Un témoignage sous X", "base_tag": "BODY", "refined_tag": "BODY"},
                {"base_text": "Envoyé depuis mon téléphone Orange", "base_tag": "FOOTER", "refined_tag": "FOOTER"},
            ],
            id="Edge case where a line ends with an isolated character",
        ),
        pytest.param(
            "     ??\n  !??!",
            [
                {"base_text": "??!??!", "base_tag": "BODY", "refined_tag": "BODY"},
            ],
            id="Edge case where the two first lines are missing word characters",
        ),
        (
            "Bonjour Mme X,\nSuite a blh blah blah.\n"
            "Bien cordialement\nJane Dupond\n"
            "(See attached file: flex.jpg)",
            [
                {"base_text": "Bonjour Mme X,", "base_tag": "HELLO", "refined_tag": "HELLO"},
                {"base_text": "Suite a blh blah blah.", "base_tag": "BODY", "refined_tag": "BODY"},
                {"base_text": "Bien cordialement", "base_tag": "GREETINGS", "refined_tag": "GREETINGS"},
                {"base_text": "Jane Dupond", "base_tag": "BODY", "refined_tag": "SIGNATURE_NAME"},
                {"base_text": "(See attached file: flex.jpg)", "base_tag": "PJ", "refined_tag": "PJ"},
            ],
        ),
        (
            "\nChère Madame\n\nC'est bien noté, merci\nBien reçu\nJ.Smith\n\n",
            [
                {"base_text": "Chère Madame", "base_tag": "HELLO", "refined_tag": "HELLO"},
                {"base_text": "C'est bien noté, merci", "base_tag": "THANKS", "refined_tag": "THANKS"},
                {"base_text": "Bien reçu", "base_tag": "BODY", "refined_tag": "BODY"},
                {"base_text": "J.Smith", "base_tag": "BODY", "refined_tag": "SIGNATURE_NAME"},
            ],
        ),
        (
            "\nBonjour Monsieur, ceci n'est pas un hello\nBonne fin de journee\nsalutations",
            [
                {"base_text": "Bonjour Monsieur, ceci n'est pas un hello", "base_tag": "BODY", "refined_tag": "BODY"},
                {"base_text": "Bonne fin de journee", "base_tag": "HELLO", "refined_tag": "HELLO"},
                {"base_text": "salutations", "base_tag": "GREETINGS", "refined_tag": "GREETINGS"},
            ],
        ),
        (
            "\nBonjour Monsieur Stanislas von den hoeggenboord\n\nbien à toi\nJ.  Smith\nChargé de clientèle",
            [
                {
                    "base_text": "Bonjour Monsieur Stanislas von den hoeggenboord",
                    "base_tag": "HELLO",
                    "refined_tag": "HELLO",
                },
                {"base_text": "bien à toi", "base_tag": "GREETINGS", "refined_tag": "GREETINGS"},
                {"base_text": "J. Smith", "base_tag": "BODY", "refined_tag": "SIGNATURE_NAME"},
                {"base_text": "Chargé de clientèle", "base_tag": "SIGNATURE", "refined_tag": "SIGNATURE"},
            ],
        ),
        (
            (
                "\n1 rdv à 18h\n\n2 ème message laissé à la locataire\n3je m'en vais au bois\n"
                "4 allée des iris\n 5bis rue Patrick Sebastien\n6-8 cours mirabeau\n 7 ter place du dahu\n"
                "8 de la rue très longue qui ne doit pas être taggée signature"
            ),
            [
                {"base_text": "1 rdv à 18h", "base_tag": "BODY", "refined_tag": "BODY"},
                {"base_text": "2 ème message laissé à la locataire", "base_tag": "BODY", "refined_tag": "BODY"},
                {"base_text": "3je m'en vais au bois", "base_tag": "BODY", "refined_tag": "BODY"},
                {"base_text": "4 allée des iris", "base_tag": "SIGNATURE", "refined_tag": "SIGNATURE"},
                {"base_text": "5bis rue Patrick Sebastien", "base_tag": "SIGNATURE", "refined_tag": "SIGNATURE"},
                {"base_text": "6-8 cours mirabeau", "base_tag": "SIGNATURE", "refined_tag": "SIGNATURE"},
                {"base_text": "7 ter place du dahu", "base_tag": "SIGNATURE", "refined_tag": "SIGNATURE"},
                {
                    "base_text": "8 de la rue très longue qui ne doit pas être taggée signature",
                    "base_tag": "BODY",
                    "refined_tag": "BODY",
                },
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
                {"base_text": "à L'attention de M Bob,", "base_tag": "FOOTER", "refined_tag": "FOOTER"},
                {"base_text": "Bonjour,", "base_tag": "HELLO", "refined_tag": "HELLO"},
                {
                    "base_text": "Je vous informe que je vais accepter la proposition de L , à savoir le paiement d'une indemnité forfaitaire de résiliation du CCMI de 4000 € TTC pour clore cette affaire.",
                    "base_tag": "BODY",
                    "refined_tag": "BODY",
                },
                {"base_text": "Cordialement.", "base_tag": "GREETINGS", "refined_tag": "GREETINGS"},
                {"base_text": "Bob Smith", "base_tag": "BODY", "refined_tag": "SIGNATURE_NAME"},
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
                {"base_text": "Monsieur Bob Smith", "base_tag": "HELLO", "refined_tag": "HELLO"},
                {
                    "base_text": "Adresse mail : BobSmith90@gmail.com",
                    "base_tag": "SIGNATURE",
                    "refined_tag": "SIGNATURE",
                },
                {"base_text": "Lucy Ange", "base_tag": "BODY", "refined_tag": "SIGNATURE_NAME"},
                {"base_text": "Bonjour Monsieur,", "base_tag": "HELLO", "refined_tag": "HELLO"},
                {
                    "base_text": "Suite à notre entretien téléphonique de ce matin, et au message que vous m'avez envoyé sur ma messagerie, je voudrais effectuer la réparation du véhicule Renault Twingo dans un garage partenaire de la Maif situé, si c'est possible.",
                    "base_tag": "BODY",
                    "refined_tag": "BODY",
                },
                {
                    "base_text": "Dans l'attente de votre réponse et en vous remerciant par avance,",
                    "base_tag": "BODY",
                    "refined_tag": "BODY",
                },
                {"base_text": "Monsieur Bob Smith", "base_tag": "HELLO", "refined_tag": "HELLO"},
                {"base_text": "Envoyé à partir de", "base_tag": "FOOTER", "refined_tag": "FOOTER"},
                {"base_text": "Courrier", "base_tag": "FOOTER", "refined_tag": "FOOTER"},
                {"base_text": "pour Windows", "base_tag": "FOOTER", "refined_tag": "FOOTER"},
                {"base_text": "Sans virus.", "base_tag": "FOOTER", "refined_tag": "FOOTER"},
                {"base_text": "www.avast.com", "base_tag": "FOOTER", "refined_tag": "FOOTER"},
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
                {"base_text": "Bob Smith", "base_tag": "BODY", "refined_tag": "SIGNATURE_NAME"},
                {"base_text": "A l’attention de Madame Lucy Ange,", "base_tag": "FOOTER", "refined_tag": "FOOTER"},
                {"base_text": "Bonjour Madame Ange,", "base_tag": "HELLO", "refined_tag": "HELLO"},
                {"base_text": "J’espère que vous allez bien.", "base_tag": "BODY", "refined_tag": "BODY"},
                {
                    "base_text": "Pour faire suite à mon mail du 21 février 2023, je me permets de revenir vers vous pour avoir votre avis sur le devis que j’ai demandé auprès d’un enquêteur.",
                    "base_tag": "BODY",
                    "refined_tag": "BODY",
                },
                {"base_text": "Voici son retour :", "base_tag": "BODY", "refined_tag": "BODY"},
                {"base_text": "Qu’en pensez-vous svp ?", "base_tag": "BODY", "refined_tag": "BODY"},
                {
                    "base_text": "Je reste à votre disposition pour tout complément d’information et vous remercie de l’intérêt que vous porterez à ma demande,",
                    "base_tag": "BODY",
                    "refined_tag": "BODY",
                },
                {"base_text": "Bien Cordialement,", "base_tag": "GREETINGS", "refined_tag": "GREETINGS"},
                {"base_text": "Bob Smith", "base_tag": "BODY", "refined_tag": "SIGNATURE_NAME"},
                {"base_text": "Tél.", "base_tag": "SIGNATURE", "refined_tag": "SIGNATURE"},
                {"base_text": "06.83.22.95.94", "base_tag": "SIGNATURE", "refined_tag": "SIGNATURE"},
            ],
        ),
        pytest.param(
            (
                "cordialement\nContact e-mail\n\n\nContact téléphone\n\n01 23 45 67 89 / abcabc@hotmail.fr\n"
                "Torroella de Montgri, le 5 avril 2023\nLes formats de fichiers acceptés sont : PDF, DOC, DOCX, JPEG, "
                "JPG, TIFF, TXT, ODT, XLS, XLSX\nTout autre format de fichiers ne sera pas transmis au dossier"
            ),
            [
                {"base_text": "cordialement", "base_tag": "GREETINGS", "refined_tag": "GREETINGS"},
                {"base_text": "Contact e-mail", "base_tag": "SIGNATURE", "refined_tag": "SIGNATURE"},
                {"base_text": "Contact téléphone", "base_tag": "SIGNATURE", "refined_tag": "SIGNATURE"},
                {
                    "base_text": "01 23 45 67 89 / abcabc@hotmail.fr",
                    "base_tag": "SIGNATURE",
                    "refined_tag": "SIGNATURE",
                },
                {
                    "base_text": "Torroella de Montgri, le 5 avril 2023",
                    "base_tag": "SIGNATURE",
                    "refined_tag": "SIGNATURE",
                },
                {
                    "base_text": "Les formats de fichiers acceptés sont : PDF, DOC, DOCX, JPEG, JPG, TIFF, TXT, ODT, XLS, XLSX",
                    "base_tag": "FOOTER",
                    "refined_tag": "FOOTER",
                },
                {
                    "base_text": "Tout autre format de fichiers ne sera pas transmis au dossier",
                    "base_tag": "FOOTER",
                    "refined_tag": "FOOTER",
                },
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
                {"base_text": "bonjour", "base_tag": "HELLO", "refined_tag": "HELLO"},
                {"base_text": "mon body", "base_tag": "BODY", "refined_tag": "BODY"},
                {"base_text": "J. Smith", "base_tag": "BODY", "refined_tag": "SIGNATURE_NAME"},
                {"base_text": "01 23 45 67 89", "base_tag": "SIGNATURE", "refined_tag": "SIGNATURE"},
                {"base_text": "Secrétaire en charge des avions", "base_tag": "SIGNATURE", "refined_tag": "SIGNATURE"},
                {
                    "base_text": "Business Analyst – Tribu Sinistres – Squad Flux Entrants",
                    "base_tag": "SIGNATURE",
                    "refined_tag": "SIGNATURE",
                },
                {
                    "base_text": "Société nationale des chemins de fer",
                    "base_tag": "SIGNATURE",
                    "refined_tag": "SIGNATURE",
                },
                {"base_text": "Conseiller MAIF", "base_tag": "SIGNATURE", "refined_tag": "SIGNATURE"},
                {"base_text": "Gestionnaire sinistre - C99G", "base_tag": "SIGNATURE", "refined_tag": "SIGNATURE"},
                {"base_text": "Service des lettres anonymes", "base_tag": "SIGNATURE", "refined_tag": "SIGNATURE"},
                {
                    "base_text": "Technicienne de gestion - EQUIPE ABC",
                    "base_tag": "SIGNATURE",
                    "refined_tag": "SIGNATURE",
                },
            ],
            id="signature_jobs",
        ),
        pytest.param(
            (
                "bonjour\nmon body\nCordialement\n\n"
                "analyste -------------------------------------- test test test test test test test\n"
            ),
            [
                {"base_text": "bonjour", "base_tag": "HELLO", "refined_tag": "HELLO"},
                {"base_text": "mon body", "base_tag": "BODY", "refined_tag": "BODY"},
                {"base_text": "Cordialement", "base_tag": "GREETINGS", "refined_tag": "GREETINGS"},
                {
                    "base_text": "analyste -------------------------------------- test test test test test test test",
                    "base_tag": "BODY",
                    "refined_tag": "BODY",
                },
            ],
            id="check_catastrophic_backtracking",
        ),
    ],
)
def test_tag_text_generic(text, expected_tags):
    # Arrange
    tagger = ContentTagger()
    refined_tagger = RefinedTagger()

    # Act
    base_tags = tagger.tag_text(text)
    refined_tags = refined_tagger.post_process_tags(base_tags)

    # Assert
    for tag_data in expected_tags:
        if "base_tag_list" not in tag_data:
            tag_data["base_tag_list"] = [tag_data["base_tag"]]
    assert refined_tags == expected_tags


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
                {"base_text": "Merci", "base_tag": "THANKS"},
                {"base_text": "Je vous remercie", "base_tag": "THANKS"},
                {"base_text": "Merci d'avance", "base_tag": "THANKS"},
                {"base_text": "Je vous remercie par avance", "base_tag": "THANKS"},
                {"base_text": "Vous en remerciant par avance.", "base_tag": "THANKS"},
            ],
            id="french thanks patterns",
        ),
    ],
)
def test_tag_text_french(text, expected_tags):
    # Arrange
    tagger = ContentTagger()

    # Act
    output_tags = tagger.tag_text(text)

    # Assert
    for tag_data in expected_tags:
        if "base_tag_list" not in tag_data:
            tag_data["base_tag_list"] = [tag_data["base_tag"]]
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
                {"base_text": "Thank you so much", "base_tag": "THANKS"},
                {"base_text": "thanks", "base_tag": "THANKS"},
                {"base_text": "thx Joanna", "base_tag": "THANKS"},
                {"base_text": "thanks but you forgot bla", "base_tag": "BODY"},
                {"base_text": "Thx however I still need the document", "base_tag": "BODY"},
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
                {"base_text": "Best", "base_tag": "GREETINGS"},
                {"base_text": "warm Wishes", "base_tag": "GREETINGS"},
                {"base_text": "regards", "base_tag": "GREETINGS"},
                {"base_text": "best regards", "base_tag": "GREETINGS"},
                {"base_text": "cheers", "base_tag": "GREETINGS"},
                {"base_text": "yours", "base_tag": "GREETINGS"},
                {"base_text": "yours truly", "base_tag": "GREETINGS"},
                {"base_text": "Sincerely", "base_tag": "GREETINGS"},
                {"base_text": "see you soon", "base_tag": "GREETINGS"},
                {"base_text": "Speak to you soon", "base_tag": "GREETINGS"},
                {"base_text": "talk soon", "base_tag": "GREETINGS"},
                {"base_text": "Take care", "base_tag": "GREETINGS"},
                {"base_text": "Catch you later", "base_tag": "GREETINGS"},
                {"base_text": "Have a fantastic day", "base_tag": "GREETINGS"},
                {"base_text": "Looking forward to your reply", "base_tag": "GREETINGS"},
                {"base_text": "I am looking forward to hearing from you", "base_tag": "GREETINGS"},
                {"base_text": "Hoping to hear from you", "base_tag": "GREETINGS"},
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
                {"base_text": "Hello John", "base_tag": "HELLO"},
                {"base_text": "hi", "base_tag": "HELLO"},
                {"base_text": "Hi there", "base_tag": "HELLO"},
                {"base_text": "good to hear from you", "base_tag": "HELLO"},
                {"base_text": "it is good to hear from you", "base_tag": "HELLO"},
                {"base_text": "I hope you are having a great week", "base_tag": "HELLO"},
                {"base_text": "how are you doing", "base_tag": "HELLO"},
                {"base_text": "how are you positioned about the matter", "base_tag": "BODY"},
                {"base_text": "i hope you are doing well", "base_tag": "HELLO"},
                {"base_text": "Good Morning Joanna", "base_tag": "HELLO"},
                {"base_text": "good Afternoon", "base_tag": "HELLO"},
                {"base_text": "Dear Jacky", "base_tag": "HELLO"},
                {"base_text": "Sir", "base_tag": "HELLO"},
                {"base_text": "Dear Madam", "base_tag": "HELLO"},
                {"base_text": "Dear Mr", "base_tag": "HELLO"},
                {"base_text": "Dear Ms.", "base_tag": "HELLO"},
                {"base_text": "Dear miss", "base_tag": "HELLO"},
                {"base_text": "Dear mrs.", "base_tag": "HELLO"},
                {"base_text": "Dear sir or madam", "base_tag": "HELLO"},
                {"base_text": "To whom it may concern", "base_tag": "HELLO"},
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
                {"base_text": "VP of Data Science", "base_tag": "SIGNATURE"},
                {"base_text": "Chief of staff", "base_tag": "SIGNATURE"},
                {"base_text": "CTO at TestMelusine", "base_tag": "SIGNATURE"},
                {"base_text": "CEOABC test", "base_tag": "BODY"},
                {"base_text": "Lead business developer", "base_tag": "SIGNATURE"},
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
                {"base_text": "9 downing street", "base_tag": "SIGNATURE"},
                {"base_text": "4-6 Beverly Hill", "base_tag": "SIGNATURE"},
                {"base_text": "4 Abbey road W24RA", "base_tag": "SIGNATURE"},
                {"base_text": "3 Ocean Rd.", "base_tag": "SIGNATURE"},
                {"base_text": "5th avenue", "base_tag": "SIGNATURE"},
                {"base_text": "221b Baker St.", "base_tag": "SIGNATURE"},
                {"base_text": "6bis River ln.", "base_tag": "SIGNATURE"},
                {"base_text": "7 Winter lane", "base_tag": "SIGNATURE"},
            ],
            id="english adsress signature patterns",
        ),
    ],
)
def test_tag_text_english(text, expected_tags):
    # Arrange
    tagger = ContentTagger()

    # Act
    output_tags = tagger.tag_text(text)

    # Assert
    for tag_data in expected_tags:
        if "base_tag_list" not in tag_data:
            tag_data["base_tag_list"] = [tag_data["base_tag"]]
    assert output_tags == expected_tags


def test_tag_list():
    # Arrange
    # Limit tags to "HELLO" and the default tag ("BODY")
    tag_list = ["HELLO"]

    # Text segment (= individual message in an email conversation)
    text = "bonjour\nblah blah blah\nmerci\ncordialement"

    # Expected tags
    expected_tags = [
        {"base_text": "bonjour", "base_tag": "HELLO"},
        {"base_text": "blah blah blah", "base_tag": "BODY"},
        {"base_text": "merci", "base_tag": "BODY"},
        {"base_text": "cordialement", "base_tag": "BODY"},
    ]

    # Instantiate and apply the Tagger
    tagger = ContentTagger(tag_list=tag_list)

    # Act
    output_tags = tagger.tag_text(text)

    # Assert
    for tag_data in expected_tags:
        if "base_tag_list" not in tag_data:
            tag_data["base_tag_list"] = [tag_data["base_tag"]]
    assert output_tags == expected_tags


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
