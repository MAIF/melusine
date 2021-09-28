import pytest
from tempfile import TemporaryDirectory
from melusine.nlp_tools.phraser_new import DeterministicPhraser


@pytest.fixture
def collocations():
    return {r"rendez[ -]vous": "rendez_vous"}


@pytest.mark.parametrize(
    "input_text, output_text",
    [
        ("rendez vous a bali", "rendez_vous a bali"),
        ("rendez-vous a bali", "rendez_vous a bali"),
        ("nada nothing rien", "nada nothing rien"),
        ("", ""),
    ],
)
def test_phraser_default(input_text, output_text, collocations):
    phraser = DeterministicPhraser(collocations=collocations)
    text = phraser.phrase(input_text)

    assert text == output_text

    with TemporaryDirectory() as tmpdir:
        phraser.save(path=tmpdir)
        phraser_reload = DeterministicPhraser.load(tmpdir)

    text = phraser_reload.phrase(input_text)
    assert text == output_text
