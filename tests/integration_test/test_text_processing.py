import pytest
import pandas as pd

from melusine.nlp_tools.text_processor import make_tokenizer_from_config
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
