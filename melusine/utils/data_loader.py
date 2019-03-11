import os
import pandas as pd


def load_email_data():
    """
    Load a pickle file containing toy email data

    Return
    ------
    pandas.DataFrame
        DataFrame with toy email data
    """

    # Data file name
    email_data_file = 'emails.pickle'

    # Path to utils folder
    utils_path = os.path.join(os.path.dirname(__file__))

    # Path to melusine (inner) folder
    melusine_path = os.path.dirname(utils_path)

    # Path to melusine (outer) folder
    project_path = os.path.dirname(melusine_path)

    # Relative path to data folder
    email_data_relative_path = 'tutorial/data/'

    # Absolute path to data folder
    email_data_absolute_path = os.path.join(project_path, email_data_relative_path)

    # Return data file
    return pd.read_pickle(email_data_absolute_path + email_data_file)
