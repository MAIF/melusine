import os.path as op
import pandas as pd


def load_email_data():
    """
    Load a pickle file containing toy email data

    Return
    ------
    pandas.DataFrame
        DataFrame with toy email data
    """

    data_directory = op.dirname(op.abspath(__file__))
    email_data_path = op.join(data_directory, 'emails.pickle')

    return pd.read_pickle(email_data_path)
