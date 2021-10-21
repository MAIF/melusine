import os
import pickle
import glob
import logging


pkl_suffix = ".pkl"
logger = logging.getLogger(__name__)


def get_file_path(filename, path, filename_prefix):
    # Check if path is a file
    if os.path.isfile(path):
        raise AttributeError(
            f"Provided path ({path}) should be a directory, not a file"
        )

    # Create directory if necessary
    os.makedirs(path, exist_ok=True)

    # Create full path
    full_path = os.path.join(
        path,
        (filename_prefix + "_" if filename_prefix else "") + filename,
    )
    return full_path


def save_pkl_generic(obj, filename, path, filename_prefix):
    # Add pickle suffix if necessary
    if not filename.endswith(pkl_suffix):
        filename += pkl_suffix

    full_path = get_file_path(filename, path, filename_prefix)

    with open(full_path, "wb") as f:
        pickle.dump(obj, f)

    return full_path


def load_pkl_generic(filename, path, filename_prefix: str = None):
    # Add pickle suffix if necessary
    if not filename.endswith(pkl_suffix):
        filename += pkl_suffix

    # Get filepath
    filepath = os.path.join(
        path,
        (filename_prefix + "_" if filename_prefix else "") + filename,
    )

    # Check file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError()

    # Load pickle file
    with open(filepath, "rb") as f:
        instance = pickle.load(f)

    return instance
