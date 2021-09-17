import os.path as op
import pandas as pd


def load_email_data(type="raw"):
    """
    Load a csv file containing toy email data

    Return
    ------
    pandas.DataFrame
        DataFrame with toy email data
    """

    data_directory = op.dirname(op.abspath(__file__))
    if type == "raw":
        email_data_path = op.join(data_directory, "emails.csv")
    elif type == "preprocessed":
        email_data_path = op.join(data_directory, "emails_preprocessed.csv")
    elif type == "full":
        email_data_path = op.join(data_directory, "emails_full.csv")
    else:
        raise AttributeError(
            f"Unknown data type {type}. Choose between 'raw', 'preprocessed' and 'full'"
        )

    return pd.read_csv(email_data_path, encoding="utf-8").fillna("")
