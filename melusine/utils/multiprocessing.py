import numpy as np
import pandas as pd
from sklearn.externals import joblib


def _apply_df(args):
    """Apply a function along an axis of the DataFrame"""
    df, func, kwargs = args
    return df.apply(func, **kwargs)


def apply_by_multiprocessing(df, func, n_jobs=1, **kwargs):
    """Apply a function along an axis of the DataFrame using multiprocessing.
    A maximum of half of the core available in the system will be used.

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
    result = joblib.Parallel(n_jobs=n_jobs, prefer='processes')(
        joblib.delayed(d.apply)(func, **kwargs) for d in np.array_split(df, n_jobs)
    )
    return pd.concat(list(result))
