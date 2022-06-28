import pytest
from melusine.nlp_tools.token_flagger import FlashtextTokenFlagger


@pytest.fixture
def token_flags():
    return {"flag_name": ["bob", "joe"], "flag_country": ["france", "usa"]}


@pytest.mark.parametrize(
    "input_tokens, output_tokens",
    [
        (["poney", "joe", "usa"], ["poney", "flag_name", "flag_country"]),
        (["france", "usa", "renard"], ["flag_country", "flag_country", "renard"]),
        (["bob"], ["flag_name"]),
        ([], []),
    ],
)
def test_token_flagger_default(input_tokens, output_tokens, token_flags):
    token_flagger = FlashtextTokenFlagger(token_flags=token_flags)
    tokens = token_flagger.flag_tokens(input_tokens)

    assert tokens == output_tokens
