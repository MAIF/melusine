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
