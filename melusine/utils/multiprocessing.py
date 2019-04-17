import numpy as np
import pandas as pd
from sklearn.externals.joblib import Parallel, delayed


def _apply_df(args):
    """Apply a function along an axis of the DataFrame"""
    df, func, kwargs = args
    return df.apply(func, **kwargs)


def apply_by_multiprocessing(df, func, n_jobs=1, **kwargs):
    """Apply a function along an axis of the DataFrame using multiprocessing.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame where the function is applied

    func : function to apply

    Returns
    -------
    pd.DataFrame
        Returns the DataFrame with the function applied.
    """
    result = Parallel(n_jobs=n_jobs, prefer='processes')(
        delayed(apply_on_chunk)(d, func, **kwargs) for d in np.array_split(df, n_jobs)
    )
    return pd.concat(list(result))

def apply_on_chunk(X, func, **kwargs):
    if "progress_bar" in kwargs:
        progress_bar = kwargs.pop('progress_bar')
    else:
        progress_bar = False
    X = apply_df(X, func, progress_bar)
    return X

def apply_df(df, func, progress_bar):
    if progress_bar:
        tqdm.pandas(leave=False, desc=func.__name__, unit='emails', dynamic_ncols=True, mininterval=2.0)
        df = df.progress_apply(func, axis=1)
    else:
        df = df.apply(func, axis=1)
    return df
