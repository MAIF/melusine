from tempfile import TemporaryDirectory

import pytest
from melusine.nlp_tools.tokenizer import RegexTokenizer


@pytest.mark.parametrize(
    "input_text, output_tokens",
    [
        ("le petit chat", ["petit", "chat"]),
        ("comme un grand", ["comme", "grand"]),
        ("le un et je", []),
    ],
)
def test_regex_tokenizer(input_text, output_tokens):
    tokenizer = RegexTokenizer(
        tokenizer_regex=r"\w+(?:[\?\-\"_]\w+)*", stopwords=["le", "un", "et", "je"]
    )

    tokens = tokenizer.tokenize(input_text)

    assert tokens == output_tokens

    # with TemporaryDirectory() as tmpdir:
    #     tokenizer.save(path=tmpdir)
    #     tokenizer_reload = RegexTokenizer.load(tmpdir)

    # tokens = tokenizer_reload.tokenize(input_text)
    # assert tokens == output_tokens
