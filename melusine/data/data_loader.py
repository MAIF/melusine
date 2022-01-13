import os.path as op
import pandas as pd


def load_email_data(type: str = "raw") -> pd.DataFrame:
    """
    Load a file containing toy email data.
    Possible types are:
    - raw : minimal DataFrame with email data
    - preprocessed : DataFrame with preprocessed email data
    - full : Full DataFrame with all email features

    Return
    ------
    pandas.DataFrame
        DataFrame with toy email data
    """

    data_directory = op.dirname(op.abspath(__file__))
    if type == "raw":
        email_data_path = op.join(data_directory, "emails.csv")
        df = pd.read_csv(email_data_path, encoding="utf-8").fillna("")
    elif type == "preprocessed":
        email_data_path = op.join(data_directory, "emails_preprocessed.pkl")
        df = pd.read_pickle(email_data_path).fillna("")
    elif type == "full":
        email_data_path = op.join(data_directory, "emails_full.pkl")
        df = pd.read_pickle(email_data_path).fillna("")
    else:
        raise AttributeError(
            f"Unknown data type {type}. Choose between 'raw', 'preprocessed' and 'full'"
        )

    return df
