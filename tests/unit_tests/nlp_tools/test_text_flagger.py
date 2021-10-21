import pytest
from tempfile import TemporaryDirectory
from melusine.nlp_tools.text_flagger import DeterministicTextFlagger


@pytest.fixture
def text_flags():
    return {r"\d{10}": "flag_phone", r"\w+@\w+\.\w{2,4}": "flag_email"}


@pytest.mark.parametrize(
    "input_text, output_text",
    [
        ("appelle moi au 0606060606", "appelle moi au flag_phone"),
        ("ecris moi a l'adresse test@domain.com", "ecris moi a l'adresse flag_email"),
        ("nada nothing rien", "nada nothing rien"),
        ("", ""),
    ],
)
def test_text_flagger_default(input_text, output_text, text_flags):
    text_flagger = DeterministicTextFlagger(text_flags=text_flags)
    text = text_flagger.flag_text(input_text)

    assert text == output_text

    with TemporaryDirectory() as tmpdir:
        text_flagger.save(path=tmpdir)
        text_flagger_reload = DeterministicTextFlagger.load(tmpdir)

    text = text_flagger_reload.flag_text(input_text)
    assert text == output_text
