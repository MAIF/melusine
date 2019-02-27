import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool


def _apply_df(args):
    """Apply a function along an axis of the DataFrame"""
    df, func, kwargs = args
    if 'progress_bar' not in kwargs:
        progress_bar = False
    else:
        progress_bar = kwargs.pop('progress_bar')

    if progress_bar:
        tqdm.pandas(leave=False, desc=func.__name__, ncols=100, unit='emails')
        return df.progress_apply(func, **kwargs)
    return df.apply(func, **kwargs)


def apply_by_multiprocessing(df, func, **kwargs):
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
        Returns the DataFrame with the funtion applied.
    """
    # define the number of cores to work with
    workers = kwargs.pop('workers')
    workers = min(workers, int(df.shape[0] / 2))
    workers = max(workers, 1)
    if df.shape[0] == 1:
        return _apply_df((df, func, kwargs))

    pool = Pool(processes=workers)
    result = pool.map(_apply_df, [(d, func, kwargs)
                                  for d in np.array_split(df, workers)])
    pool.close()
    return pd.concat(list(result))
