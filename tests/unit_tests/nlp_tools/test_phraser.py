import pytest
import pandas as pd
from melusine.nlp_tools.phraser import Phraser


@pytest.mark.parametrize(
    "input_tokens, output_tokens",
    [
        (["rendez", "vous", "ici"], ["rendez_vous", "ici"]),
        (["rendez", "vous", "la", "bas"], ["rendez_vous", "la", "bas"]),
        (["hello"], ["hello"]),
        ([], []),
    ],
)
def test_phraser(input_tokens, output_tokens):
    df = pd.DataFrame(
        {
            "tokens": [
                ["un", "rendez", "vous", "cool"],
                ["le", "rendez", "vous"],
                ["rendez", "vous", "client"],
                ["rendez", "vous", "ici"],
                ["rendez", "vous", "la", "bas"],
                [],
            ]
        }
    )
    phraser = Phraser(threshold=1, min_count=1)
    phraser.fit(df)

    tokens = phraser.phraser_[input_tokens]
    assert tokens == output_tokens
