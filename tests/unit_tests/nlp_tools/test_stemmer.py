import pytest
from melusine.nlp_tools.stemmer import Stemmer


@pytest.mark.parametrize(
    "input_tokens, output_tokens",
    [
        (["envoye", "courrier"], ["envoy", "courri"]),
        (["semblerait", "trouver"], ["sembl", "trouv"]),
    ],
)
def test_stemmer(input_tokens, output_tokens):

    stemmer = Stemmer()

    stemm = stemmer._stemming(input_tokens)

    assert stemm == output_tokens