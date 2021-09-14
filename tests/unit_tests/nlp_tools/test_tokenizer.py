import pytest
from melusine.nlp_tools.tokenizer import WordLevelTokenizer


@pytest.mark.parametrize(
    "input_text, expected_tokens",
    [
        ("hello,   world!", ["hello", "world"]),
        ("\n   hello world    ", ["hello", "world"]),
        ("hello(world)", ["hello", "world"]),
        ("----- hello\tworld *****", ["hello", "world"]),
        ("hello.world", ["hello", "world"]),
        ("hello!world", ["hello", "world"]),
        ("hello\nworld", ["hello", "world"]),
    ],
)
def test_tokenizer_default(input_text, expected_tokens):
    tokenizer = WordLevelTokenizer()
    tokens = tokenizer.tokenize(input_text)

    assert tokens == expected_tokens


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
def test_tokenizer_flag_items(input_text, expected_text):
    tokenizer = WordLevelTokenizer()
    text = tokenizer._flag_text(input_text)

    assert text == expected_text
