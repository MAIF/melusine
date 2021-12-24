import pytest
from melusine.nlp_tools.normalizer import Normalizer


@pytest.mark.parametrize(
    "input_text, lowercase, output_text",
    [
        ("Héllö WORLD", True, "hello world"),
        ("Hèllo WÖRLD", False, "Hello WORLD"),
        ("", False, ""),
    ],
)
def test_text_flagger_default(input_text, lowercase, output_text):
    normalizer = Normalizer(lowercase=lowercase)
    text = normalizer.normalize(input_text)

    assert text == output_text
