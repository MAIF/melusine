import os.path as op

import pandas as pd


def load_email_data(type: str = "raw") -> pd.DataFrame:
    """
    Function to load a file containing toy email data.
    Possible types are:
    - raw : minimal DataFrame with email data
    - preprocessed : DataFrame with preprocessed email data
    - full : Full DataFrame with all email features

    Return
    ------
    pandas.DataFrame
        DataFrame with toy email data
    """

    # Path to data directory
    data_directory = op.dirname(op.abspath(__file__))

    # Load raw data
    if type == "raw":
        email_data_path = op.join(data_directory, "emails.json")
        df = pd.read_json(email_data_path, orient="records").fillna("")

    # Load preprocessed data
    elif type == "preprocessed":
        email_data_path = op.join(data_directory, "emails_preprocessed.json")
        df = pd.read_json(email_data_path, orient="records").fillna("")

    # Load preprocessed data with feature engineering
    elif type == "full":
        email_data_path = op.join(data_directory, "emails_full.json")
        df = pd.read_json(email_data_path, orient="records").fillna("")
    else:
        raise ValueError(f"Unknown data type {type}. Choose between 'raw', 'preprocessed' and 'full'")

    return df
