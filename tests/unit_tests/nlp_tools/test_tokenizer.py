import pytest
import pandas as pd
from melusine.nlp_tools.tokenizer import Tokenizer


@pytest.mark.parametrize(
    "input_df, expected_serie",
    [
        (
            pd.DataFrame(
                ["Bonjour, je m'appelle Nicolas : nicolas_nom@hotmail.fr"],
                columns=["body"],
            ),
            pd.Series(
                [
                    "Bonjour",
                    "appelle",
                    "flag_name_",
                    "nicolas_nom",
                    "hotmail",
                    "fr",
                ]
            ),
        ),
        (
            pd.DataFrame(
                [
                    "Bonjour, je fais suite au devis réalisé pour le contrat en mai dernier. "
                    "Cdlt, Jean-pierre."
                ],
                columns=["body"],
            ),
            pd.Series(
                [
                    "Bonjour",
                    "fais",
                    "suite",
                    "devis",
                    "réalisé",
                    "contrat",
                    "mai",
                    "dernier",
                    "Cdlt",
                    "flag_name_",
                ]
            ),
        ),
    ],
)
def test_flag_name(input_df, expected_serie):
    tokenizer = Tokenizer(input_column="body")
    result = pd.Series(tokenizer.transform(input_df).loc[0]["tokens"])
    pd.testing.assert_series_equal(result, expected_serie)
