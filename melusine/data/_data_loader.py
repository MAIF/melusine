from pathlib import Path

import pandas as pd


def load_email_data() -> pd.DataFrame:
    """
    Function to load a file containing toy email data.

    Return
    ------
    pandas.DataFrame
        DataFrame with toy email data
    """

    # Path to data directory
    data_directory = Path(__file__).parent.resolve()

    # Load data
    email_data_path = data_directory / "emails.json"
    df = pd.read_json(email_data_path, orient="records").fillna("")

    return df
