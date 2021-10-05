import pytest
import pandas as pd

from melusine.nlp_tools.text_processor import TextProcessor, make_tokenizer_from_config
from melusine import config


@pytest.fixture
def default_tokenizer():
    return make_tokenizer_from_config(config)


@pytest.mark.parametrize(
    "input_text, expected_tokens",
    [
        ("Hello,   world!", ["hello", "world"]),
        ("\n   hello world    ", ["hello", "world"]),
        ("hello(world)", ["hello", "world"]),
        ("----- hello\tworld *****", ["hello", "world"]),
        ("hello.world", ["hello", "world"]),
        ("hello!World", ["hello", "world"]),
        ("hello\nworld", ["hello", "world"]),
        (
            "bonjour! je m'appelle roger, mon numéro est 0600000000",
            ["bonjour", "appelle", "flag_name_", "numero", "flag_phone_"],
        ),
        (
            "j'ai rendez vous avec simone",
            ["rendez_vous", "flag_name_"],
        ),
    ],
)
def test_tokenizer_default(default_tokenizer, input_text, expected_tokens):
    df = pd.DataFrame({"text": [input_text]})
    df = default_tokenizer.transform(df)
    tokens = df["tokens"].iloc[0]

    assert tokens == expected_tokens


if False:

    @pytest.mark.parametrize(
        "input_text, expected_text",
        [
            (
                "Bonjour, mon email : prenom.nom@hotmail.fr",
                "Bonjour, mon email :  flag_mail_ ",
            ),
            ("Mon numéro : 01.23.45.67.89", "Mon numéro :  flag_phone_ "),
            ("01 23 45 67 89 et 01.23.45.67.89", " flag_phone_  et  flag_phone_ "),
            ("mon numéro01 23 45 67 89", "mon numéro flag_phone_ "),
            (
                "le montant du contrat est du 18000$, soit 17000euros",
                "le montant du contrat est du  flag_amount_ , soit  flag_amount_ ",
            ),
            (
                "J'habite au 1 rue de la paix, Paris 75002",
                "J'habite au 1 rue de la paix, Paris  flag_cp_ ",
            ),
            (
                "Rendez-vous le 18 décembre 2019 ou le 19/12/19 ou le 20.12.19 à 14h30",
                "Rendez-vous le  flag_date_  ou le  flag_date_  ou le  flag_date_  à  flag_time_ ",
            ),
            (
                "le 14/12 tu me devras 20.05 dollars",
                "le  flag_date_  tu me devras  flag_amount_ ",
            ),
        ],
    )
    def test_tokenizer_flag_items(default_tokenizer, input_text, expected_text):
        text = default_tokenizer._flag_text(input_text)

        assert text == expected_text

    @pytest.mark.parametrize(
        "input_tokens, output_tokens",
        [
            (["poney", "carton", "amelie"], ["poney", "carton", "flag_name_"]),
            (["chat", "jose", "renard"], ["chat", "flag_name_", "renard"]),
            (["charlotte"], ["flag_name_"]),
        ],
    )
    def test_tokenizer_flag_names(default_tokenizer, input_tokens, output_tokens):
        tokenizer = TextProcessor()
        tokens = default_tokenizer._flag_names(input_tokens)

        assert tokens == output_tokens

    @pytest.mark.parametrize(
        "input_tokens, output_tokens",
        [
            (["le", "petit", "chat"], ["petit", "chat"]),
            (["comme", "un", "grand"], ["comme", "grand"]),
            (["le", "un", "et", "je"], []),
        ],
    )
    def test_tokenizer_remove_stopwords(input_tokens, output_tokens):
        tokenizer = TextProcessor()
        tokens = tokenizer._remove_stopwords(input_tokens)

        assert tokens == output_tokens

    @pytest.mark.parametrize(
        "input_text, expected_text",
        [
            (
                "rendez vous a bali",
                "rendez_vous a bali",
            ),
        ],
    )
    def test_tokenizer_join_collocations(input_text, expected_text):
        tokenizer = TextProcessor()
        text = tokenizer._flag_text(input_text, tokenizer.collocations_dict)

        assert text == expected_text

    @pytest.mark.parametrize(
        "input_text, expected_text",
        [("éàèùöï", "eaeuoi")],
    )
    def test_tokenizer_normalize_text(input_text, expected_text):
        tokenizer = TextProcessor()
        text = tokenizer._normalize_text(input_text)

        assert text == expected_text

    @pytest.mark.parametrize(
        "input_text, lowercase, expected_tokens",
        [
            ("Hello WORLD", True, ["hello", "world"]),
            ("Hello WORLD", False, ["Hello", "WORLD"]),
        ],
    )
    def test_tokenizer_normalize_text(input_text, lowercase, expected_tokens):
        tokenizer = TextProcessor(lowercase=lowercase)
        tokens = tokenizer.process(input_text)

        assert tokens == expected_tokens
