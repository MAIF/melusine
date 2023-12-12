"""
Unit test for processors.py
"""
import pytest

from melusine.processors import (
    DateProcessor,
    DeterministicTextFlagger,
    Message,
    Normalizer,
    RegexTokenizer,
    Segmenter,
    TextExtractor,
    TokensExtractor,
    TransferredEmailProcessor,
)


@pytest.mark.parametrize(
    "input_text, lowercase, output_text",
    [
        ("Héllö WORLD", True, "hello world"),
        ("Hèllo WÖRLD", False, "Hello WORLD"),
        ("", False, ""),
    ],
)
def test_normalizer(input_text, lowercase, output_text):
    """Unit test"""

    normalizer = Normalizer(lowercase=lowercase)
    text = normalizer.normalize_text(input_text)
    assert text == output_text

    assert normalizer.normalize_text(1.25) == ""


def test_normalizer_messages():
    """Unit test"""

    message_1 = Message(meta="", text="Héllö WORLD")
    message_2 = Message(meta="abcd", text="Héllö heLLo")
    normalizer = Normalizer(lowercase=True, input_columns="messages")
    message_list = normalizer.normalize_message([message_1, message_2])

    assert message_list[0].clean_text == "hello world"
    assert message_list[1].clean_text == "hello hello"


@pytest.mark.parametrize(
    "input_text, output_tokens, lowercase, normalization_form",
    [
        ("le petit Chat", ["petit", "chat"], True, None),
        ("le Géant", ["géant"], True, None),
        ("le Géant", ["Geant"], False, "NFKD"),
        ("Comme un grand", ["Comme", "grand"], False, None),
        ("le un et je", [], False, None),
    ],
)
def test_tokenizer(input_text, output_tokens, lowercase, normalization_form):
    """Unit test"""

    tokenizer = RegexTokenizer(
        tokenizer_regex=r"\w+(?:[\?\-\"_]\w+)*",
        stopwords=["le", "un", "et", "je"],
        lowercase=lowercase,
        normalization_form=normalization_form,
    )

    tokens = tokenizer.tokenize(input_text)
    assert tokens == output_tokens


@pytest.mark.parametrize(
    "input_text, expected_messages",
    [
        ("Hello World", [Message(meta="", text="Hello World")]),
        (
            "Merci\nDe : jean@gmail.com\nObjet: Votre attestation\nVoici l'attestation",
            [
                Message(meta="", text="Merci"),
                Message(
                    meta="De : jean@gmail.com\nObjet: Votre attestation",
                    text="Voici l'attestation",
                ),
            ],
        ),
        (
            "Merci\nObjet: Votre attestation\nVoici l'attestation",
            [
                Message(meta="", text="Merci\nObjet: Votre attestation\nVoici l'attestation"),
            ],
        ),
        (
            "Merci\nDe : jean@gmail.com\nSujet : ABCD\nObjet: Votre attestation\nVoici l'attestation",
            [
                Message(meta="", text="Merci"),
                Message(
                    meta="De : jean@gmail.com\nSujet : ABCD\nObjet: Votre attestation",
                    text="Voici l'attestation",
                ),
            ],
        ),
        (
            "Je vous ai Envoyé :\n- le devis\nla facture\nSalutations",
            [
                Message(meta="", text="Je vous ai Envoyé :\n- le devis\nla facture\nSalutations"),
            ],
        ),
        (
            "Message 1\nLe 2 févr. 2022 à 09:10,\ntest@maif.fr\na écrit :\n\n\nMessage 2",
            [
                Message(meta="", text="Message 1"),
                Message(meta="Le 2 févr. 2022 à 09:10,\ntest@maif.fr\na écrit :", text="Message 2"),
            ],
        ),
        (
            "Message 1\nmail transféré\n------------\nLe 2 févr. 2022 à 09:10,\ntest@maif.fr\na écrit :\n\n\nMessage 2",
            [
                Message(meta="", text="Message 1"),
                Message(
                    meta="mail transféré\n------------\nLe 2 févr. 2022 à 09:10,\ntest@maif.fr\na écrit :",
                    text="Message 2",
                ),
            ],
        ),
    ],
)
def test_segmenter(input_text, expected_messages):
    """Unit test"""

    segmenter = Segmenter()
    result = segmenter.segment_text(input_text)
    for i, message in enumerate(result):
        assert message.meta == expected_messages[i].meta
        assert message.text == expected_messages[i].text


@pytest.mark.parametrize(
    "input_message_list, expected_text",
    [
        (
            [
                Message(meta="", text="Hello world"),
                Message(
                    meta="",
                    text="Hello world",
                ),
            ],
            "Hello world",
        ),
        (
            [
                Message(meta="", text="Merci"),
                Message(
                    meta="De : jean@gmail.com\nObjet: Votre attestation",
                    text="Voici l'attestation",
                ),
            ],
            "Merci",
        ),
        (
            [
                Message(meta="", text="Merci", tags=[("THANKS", "Merci")]),
            ],
            "Merci",
        ),
    ],
)
def test_text_extractor(input_message_list, expected_text):
    """Unit test"""

    extractor = TextExtractor(output_columns="text", n_messages=1)
    result = extractor.extract(input_message_list)
    assert result == expected_text


def test_text_extractor_error():
    """Unit test"""
    with pytest.raises(ValueError):
        _ = TextExtractor(output_columns="text", n_messages=1, include_tags=["A"], exclude_tags=["B"])


def test_text_extractor_multiple_messages():
    """Unit test"""
    message_list = [
        Message(meta="", text="", tags=[("BODY", "A"), ("GREETINGS", "G"), ("BODY", "A")]),
        Message(meta="", text="", tags=[("BODY", "B"), ("BODY", "B"), ("BODY", "B")]),
        Message(meta="", text="", tags=[("GREETINGS", "G"), ("BODY", "C"), ("BODY", "C")]),
    ]
    expected_output = "A\nB\nB\nB"

    extractor = TextExtractor(
        output_columns="text",
        n_messages=None,
        stop_at=["GREETINGS"],
        include_tags=["BODY"],
    )
    result = extractor.extract(message_list)
    assert result == expected_output


def test_text_extractor_with_tags():
    """Unit test"""
    input_message_list = [
        Message(meta="", text="Bonjour\nblahblah\nMerci"),
        Message(meta="", text="Bonjour2\nMerci2"),
    ]
    input_message_list[0].tags = [("HELLO", "Bonjour"), ("CUSTOM_TAG", "blahblah"), ("THANKS", "Merci")]
    input_message_list[1].tags = [("HELLO", "Bonjour2"), ("THANKS", "Merci2")]

    extractor = TextExtractor(
        output_columns="text",
        exclude_tags=["HELLO", "CUSTOM_TAG"],
    )
    result = extractor.extract(input_message_list)
    assert result == "Merci"


def test_token_extractor():
    """Unit test"""
    separator = "PAD"
    pad_size = 2

    token_extractor = TokensExtractor(
        input_columns=("body_tok", "header_tok"),
        output_columns=("toks,"),
        pad_size=pad_size,
        sep_token=separator,
    )

    data = {
        "body_tok": ["my", "body"],
        "header_tok": ["my", "header"],
    }

    extracted = token_extractor.extract(data)

    assert extracted == ["my", "body", separator, separator, "my", "header"]


@pytest.mark.parametrize(
    "input_text, output_text",
    [
        ("appelle moi au 0606060606", "appelle moi au flag_phone"),
        ("Tel:0606060606", "Tel: flag_phone"),
        ("ecris moi a l'adresse test@domain.com", "ecris moi a l'adresse flag_email"),
        ("nada nothing rien", "nada nothing rien"),
        ("", ""),
    ],
)
def test_text_flagger_default(input_text, output_text):
    """Unit test"""
    text_flags = {
        "numeric_flags": {r"\d{10}": "flag_phone"},
        r"\w+@\w+\.\w{2,4}": "flag_email",
    }
    text_flagger = DeterministicTextFlagger(text_flags=text_flags)
    text = text_flagger.flag_text(input_text)

    assert text == output_text


@pytest.mark.parametrize(
    "input_text, output_text, add_spaces, remove_multiple_spaces",
    [
        ("Tel:0606060606", "Tel:flag_phone", False, True),
        ("Tel: 0606060606", "Tel: flag_phone", True, True),
        ("Tel: 0606060606", "Tel:  flag_phone", True, False),
        ("", "", False, False),
    ],
)
def test_text_flagger_args(input_text, output_text, add_spaces, remove_multiple_spaces):
    """Unit test"""
    text_flags = {
        "numeric_flags": {r"\d{10}": "flag_phone"},
    }
    text_flagger = DeterministicTextFlagger(
        text_flags=text_flags,
        add_spaces=add_spaces,
        remove_multiple_spaces=remove_multiple_spaces,
    )
    text = text_flagger.flag_text(input_text)

    assert text == output_text


def test_date_processor_instanciation():
    """Test instantiate processor"""
    _ = DateProcessor(input_columns="date", output_columns="date")


@pytest.mark.parametrize(
    "date_str, expected_iso_format",
    [
        ("jeudi 29 avril 2021 21:07", "2021-04-29"),
        ("jeudi 9 avril 2021 21:07", "2021-04-09"),
        ("mar. 11 mai 2021 à 14:29", "2021-05-11"),
        ("4 mai 2021 à 18:14:26 UTC+2", "2021-05-04"),
        ("Lundi 04 janvier 2021 11:35", "2021-01-04"),
        ("mié, 16 dic 2020 a las 16:00", "2020-12-16"),
        ("mié, 16 dic, 2020 a las 16:00", "2020-12-16"),
        ("gio 9 set 2021 alle ore 16:16", "2021-09-09"),
        ("woensdag 16 december 2020 10:01", "2020-12-16"),
        ("lundi 19 juihet 2021, 15:25:42 utc+2", "2021-07-19"),
        ("16 june 2021 14:38", "2021-06-16"),
        ("5 july 2021 at 16:49:41 cest", "2021-07-05"),
        ("sunday, 13 jun 2021", "2021-06-13"),
        ("sunday, 13 jun, 2021", "2021-06-13"),
        ("26 eki 2020 pzt 14:27", "2020-10-26"),
        ("vendredi 6 février 2019 09 h 02", "2019-02-06"),
    ],
)
def test_date_processor(date_str: str, expected_iso_format: str) -> None:
    """Simple base tests"""
    date_processor = DateProcessor()
    date_iso_format: str = date_processor.parse_date_to_iso(date_str)

    assert date_iso_format == expected_iso_format


@pytest.mark.parametrize(
    "message_list, tags_to_ignore, expected_len_message_list, expected_original_from",
    [
        # Direct transfer
        pytest.param(
            [
                Message(
                    meta="De: test.test@test.fr <test.test@test.fr>\n"
                    "Envoyé: mercredi 4 mai 2022 11:11\n"
                    "A: avocats@test.fr; BOB Morane <bob.morane@test.fr>\n"
                    "Objet: dossier Test ,\n",
                    text="bla bla bla",
                    tags=[("BODY", "bla bla bla")],
                )
            ],
            "('FOOTER', 'SIGNATURE')",
            1,
            "test.test@test.fr",
            id="direct_transfer",
        ),
        # Direct transfer with Footer
        pytest.param(
            [
                Message(
                    meta="",
                    text="Envoyé depuis mon Iphone",
                    tags=[("FOOTER", "Envoyé depuis mon Iphone")],
                ),
                Message(
                    meta="De: test.test@test.fr <test.test@test.fr>\n"
                    "Envoyé: mercredi 4 mai 2022 11:11\n"
                    "A: avocats@test.fr; BOB Morane <bob.morane@test.fr>\n"
                    "Objet: dossier Test ,\n",
                    text="bla bla bla",
                    tags=[("BODY", "bla bla bla")],
                ),
            ],
            "('FOOTER', 'SIGNATURE')",
            1,
            "test.test@test.fr",
            id="direct_transfer_with_footer",
        ),
        # Direct transfer with Signature (pattern De:\ntest@gmail\nA:\n)
        pytest.param(
            [
                Message(
                    meta="",
                    text="Jane Doe\n4 rue des oliviers 75001 Ville",
                    tags=[
                        ("SIGNATURE", "4 rue des oliviers 75001 Ville"),
                    ],
                ),
                Message(
                    meta="De :\ntest.test42@test.fr\nEnvoyé :\nvendredi 03 mars 2023 14:28\nÀ :"
                    "\nana@test.fr\nObjet :\nTEST",
                    text="bla bla bla",
                    tags=[("BODY", "bla bla bla")],
                ),
            ],
            "('FOOTER', 'SIGNATURE')",
            1,
            "test.test42@test.fr",
            id="direct_transfer_with_signature",
        ),
        # Direct transfer with multiple Signature patterns
        pytest.param(
            [
                Message(
                    meta="",
                    text="Jane Doe\n4 rue des oliviers 75001 Ville",
                    tags=[
                        ("SIGNATURE_NAME", "Jane Doe"),
                        ("SIGNATURE", "4 rue des oliviers 75001 Ville"),
                    ],
                ),
                Message(
                    meta="De :\ntest.test42@test.fr\nEnvoyé :\nvendredi 03 mars 2023 14:28\nÀ :"
                    "\nana@test.fr\nObjet :\nTEST",
                    text="bla bla bla",
                    tags=[("BODY", "bla bla bla")],
                ),
            ],
            "('FOOTER', 'SIGNATURE')",
            1,
            "test.test42@test.fr",
            id="direct_transfer_with_multiple_signatures",
        ),
        # Other transition pattern (pattern De:\nTest <test@gmail>\nA:\n)
        pytest.param(
            [
                Message(
                    meta=(
                        "De:\nANNA <42test.test@test.fr>\nDate:\n3 mars 2023 à 16:42:50 UTC+1\n"
                        "À:\nbob@test.fr\nObjet:\nTR : 1223456"
                    ),
                    text="bla bla bla",
                    tags=[("BODY", "bla bla bla")],
                ),
            ],
            "('FOOTER', 'SIGNATURE')",
            1,
            "42test.test@test.fr",
            id="transition_pattern_de_date_a",
        ),
        # Other transition pattern (pattern De: "test@gmail"\nA:)
        pytest.param(
            [
                Message(
                    meta=(
                        """De: "test_test@test.fr"\nDate:\n3 mars 2023 à 16:42:50 UTC+1\n"""
                        "À:\nbob@test.fr\nObjet:\nTR : 1223456"
                    ),
                    text="bla bla bla",
                    tags=[("BODY", "bla bla bla")],
                ),
            ],
            "('FOOTER', 'SIGNATURE')",
            1,
            "test_test@test.fr",
            id="transition_pattern_de_a",
        ),
        # Other transition pattern (pattern Le 1 mars..., Abc <abc@gmail.com> a écrit)
        pytest.param(
            [
                Message(
                    meta=("Le 2 mars 2023 à 18:18, Bob <test.test.test@test.fr> a écrit :"),
                    text="bla bla bla",
                    tags=[("BODY", "bla bla bla")],
                ),
            ],
            "('FOOTER', 'SIGNATURE')",
            1,
            "test.test.test@test.fr",
            id="transition_pattern_le_date_text_a_ecrit",
        ),
        # Other transition pattern (pattern Le 01/01/2001 12:12, abc@gmail.com a écrit)
        pytest.param(
            [
                Message(
                    meta=("Le 01/01/2001 11:14, test.test.test@test.fr a écrit :"),
                    text="bla bla bla",
                    tags=[("BODY", "bla bla bla")],
                ),
            ],
            "('FOOTER', 'SIGNATURE')",
            1,
            "test.test.test@test.fr",
            id="transition_pattern_le_date_numbers_a_ecrit",
        ),
        # Other transition pattern (pattern Le dim., 12:12, abc@gmail.com a écrit)
        pytest.param(
            [
                Message(
                    meta=("Le 01/01/2001 11:14, test.test.test@test.fr a écrit :"),
                    text="bla bla bla",
                    tags=[("BODY", "bla bla bla")],
                ),
            ],
            "('FOOTER', 'SIGNATURE')",
            1,
            "test.test.test@test.fr",
            id="transition_pattern_le_weekday_a_ecrit",
        ),
        # Test argument tags_to_ignore (removed Signature from the list)
        pytest.param(
            [
                Message(
                    meta="",
                    text="Jane Doe\n4 rue des oliviers 75001 Ville",
                    tags=[("SIGNATURE", "Jane Doe\n4 rue des oliviers 75001 Ville")],
                ),
                Message(
                    meta="De: test.test@test.fr <test.test@test.fr>\n"
                    "Envoyé: mercredi 4 mai 2022 11:11\n"
                    "A: avocats@test.fr; BOB Morane <bob.morane@test.fr>\n"
                    "Objet: dossier Test ,\n",
                    text="bla bla bla",
                    tags=[("BODY", "bla bla bla")],
                ),
            ],
            "('FOOTER',)",
            2,
            None,
            id="tags_to_ignore",
        ),
        # Not a transfer
        pytest.param(
            [
                Message(
                    meta="",
                    text="J'entends le loup, le renard et la belette",
                    tags=[("BODY", "J'entends le loup, le renard et la belette")],
                ),
                Message(
                    meta="De: test.test@test.fr <test.test@test.fr>\n"
                    "Envoyé: mercredi 4 mai 2022 11:11\n"
                    "A: avocats@test.fr; BOB Morane <bob.morane@test.fr>\n"
                    "Objet: dossier Test ,\n",
                    text="bla bla bla",
                    tags=[("BODY", "bla bla bla")],
                ),
            ],
            "('FOOTER', 'SIGNATURE')",
            2,
            None,
            id="not_a_transfer",
        ),
        # Empty mail
        pytest.param(
            [
                Message(
                    meta="",
                    text="",
                    tags=[("BODY", "")],
                ),
            ],
            "('FOOTER', 'SIGNATURE')",
            1,
            None,
            id="empty_email",
        ),
        # No email address in the meta
        pytest.param(
            [
                Message(
                    meta="",
                    text="Envoyé de mon iPhone",
                    tags=[("FOOTER", "bla 1")],
                ),
                Message(
                    meta="Nothing useful",
                    text="bla 2",
                    tags=[("BODY", "bla 2")],
                ),
            ],
            "('FOOTER', 'SIGNATURE')",
            1,
            None,
            id="missing_email_address",
        ),
    ],
)
def test_transferred_email_processor(message_list, tags_to_ignore, expected_len_message_list, expected_original_from):
    """Unit test"""
    processor = TransferredEmailProcessor(tags_to_ignore=tags_to_ignore, messages_column="message_list")
    processed_message_list, clean_from = processor.process_transfered_mail(message_list=message_list)

    assert clean_from == expected_original_from
    assert len(processed_message_list) == expected_len_message_list
