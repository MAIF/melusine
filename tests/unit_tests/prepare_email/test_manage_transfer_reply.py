import pandas as pd
import numpy as np
from melusine.prepare_email.manage_transfer_reply import (
    add_boolean_answer,
    add_boolean_transfer,
    check_mail_begin_by_transfer,
)


def test_add_boolean_answer():
    input_df = pd.DataFrame(
        {
            "header": [
                "Bonjour, je suis disponible",
                "RE: bonjour",
                "re: bonjour",
                "TR: bonjour",
                "Fwd: bonjour",
                "Re: bonjour",
                np.nan,
                "",
            ],
        }
    )

    output_df = pd.Series([False, True, True, False, False, True, False, False])

    result = input_df.apply(add_boolean_answer, axis=1)
    pd.testing.assert_series_equal(result, output_df)


def test_add_boolean_transfer():
    input_df = pd.DataFrame(
        {
            "header": [
                "Bonjour, je suis disponible",
                "RE: bonjour",
                "re: bonjour",
                "TR: bonjour",
                "Tr: bonjour",
                "Fwd: bonjour",
                "FW: bonjour",
                "FWD: bonjour",
                "Fw: bonjour",
                np.nan,
                "",
            ],
        }
    )

    output_df = pd.Series(
        [False, False, False, True, True, True, True, True, True, False, False]
    )

    result = input_df.apply(add_boolean_transfer, axis=1)
    pd.testing.assert_series_equal(result, output_df)


def test_check_mail_begin_by_transfer():
    input_df = pd.DataFrame(
        {
            "body": [
                "Bonjour, je suis disponible",
                "--- Transféré par <xxxx@gmail.com> ---- Bonjour, je suis disponible",
                "De : <xxxx@gmail.com> : salut, je suis disponible",
                np.nan,
                "",
            ]
        }
    )

    output_df = pd.Series([False, True, True, False, False])

    result = input_df.apply(check_mail_begin_by_transfer, axis=1)
    pd.testing.assert_series_equal(result, output_df)
