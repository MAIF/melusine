"""
Useful class to define its own transformer using specific functions
in a specific order to apply along a row of DataFrame (axis=1).

It is compatible with scikit-learn API (i.e. contains fit, transform methods).
"""

import pandas as pd
import copy
from sklearn.base import BaseEstimator, TransformerMixin
from melusine.utils.multiprocessing import apply_by_multiprocessing


def __check_function_type(func):
    """Check if it is a function-like object."""
    if not callable(func):
        raise TypeError(
            "First item of the tuple (func, args, cols) must be a function-like \
object not a {} object".format(
                type(func)
            )
        )
    else:
        return func


def __check_args_type(args):
    """Check if it is a tuple-like object."""
    if args is None or args == ():
        return None
    elif isinstance(args, int) or isinstance(args, str) or isinstance(args, list):
        # manage the case of 1 element in tuple (example: args=(4))
        return (args,)
    elif not isinstance(args, tuple):
        raise TypeError(
            "Second item of the tuple (func, args, cols) must be tuple-like \
object not a {} object".format(
                type(args)
            )
        )
    else:
        return args


def __check_colnames_type(cols):
    """Check if it is a list-like object."""
    if cols is None or cols == []:
        return None
    elif not isinstance(cols, list):
        raise TypeError(
            "Third item of the tuple (func, args, cols) must be list-like \
object not a {} object".format(
                type(cols)
            )
        )
    else:
        return cols


def _check_tuple(func, args=None, cols=None):
    """Complete checking of each element for the 'function_scheduler'
    parameter."""
    # check types of each parameters
    func = __check_function_type(func)
    args = __check_args_type(args)
    cols = __check_colnames_type(cols)

    return (func, args, cols)


class TransformerScheduler(BaseEstimator, TransformerMixin):
    """
    This class aims to provide a good way to define its own transformer.
    It takes a list of function defined in a specific order to apply along a
    row of DataFrame (axis=1).
    Transformer returned is compatible with scikit-learn API
    (i.e. contains fit, transform methods).

    Parameters
    ----------
    functions_scheduler : list of tuples, (function, tuple, list)
        List of function to be applied in a specific order.
        Each element of the list has to be defined as follow:
        (`function`, `argument(s) used by the function (optional)`, `colname(s)
        returned (optional)`)

    mode : str {'apply', 'apply_by_multiprocessing'}, optional
        Define mode to apply function along a row axis (axis=1).
        Default value, 'apply'.
        If set to 'apply_by_multiprocessing', it uses multiprocessing tool
        to parallelize computation.

    n_jobs : int, optional
        Number of cores used for computation. Default value, 1.

    progress_bar : boolean, optional
        Whether to print a progress bar from tqdm package. Default value, True.
        Works only when mode is set to 'apply_by_multiprocessing'.

    copy : boolean, optional
        Make a copy of DataFrame. Default value, True.

    verbose : int, optional
        Verosity mode, print loggers. Default value, 0.

    Attributes
    ----------
    function_scheduler, mode, n_jobs, progress_bar

    Examples
    --------
    >>> from melusine.utils.transformer_scheduler import TransformerScheduler

    >>> MelusineTransformer = TransformerScheduler(
    >>>     functions_scheduler=[
    >>>         (my_function_1, (argument1, argument2), ['return_col_A']),
    >>>         (my_function_2, None, ['return_col_B', 'return_col_C'])
    >>>         (my_function_3, (), ['return_col_D'])
    >>>     ])

    """

    def __init__(
        self,
        functions_scheduler,
        mode="apply",
        n_jobs=1,
        progress_bar=True,
        copy=True,
        verbose=0,
    ):
        self.functions_scheduler = functions_scheduler
        self.mode = mode
        self.n_jobs = n_jobs
        self.progress_bar = True
        self.copy = copy
        self.verbose = verbose

        # check input parameters type
        for tuple_ in functions_scheduler:
            func, args, cols = _check_tuple(*tuple_)

    def fit(self, X, y=None):
        """Unused method. Defined only for compatibility with scikit-learn API."""
        return self

    def transform(self, X):
        """Apply functions defined in the `function_scheduler` parameter.

        Parameters
        ----------
        X : pandas.DataFrame,
            Data on which transformations are applied.

        Returns
        -------
        pandas.DataFrame
        """
        # Case input is a dict
        if isinstance(X, dict):
            if self.copy:
                X_ = copy.deepcopy(X)
            else:
                X_ = X

            apply_func = self.apply_dict

        # Case input is a DataFrame
        else:
            if self.copy:
                X_ = X.copy()
            else:
                X_ = X

            # Multiprocessing (or progress bar or both)
            if self.mode == "apply_by_multiprocessing":
                apply_func = self.apply_pandas_multiprocessing

            # Single process (no progress bar)
            else:
                apply_func = self.apply_pandas

        for tuple_ in self.functions_scheduler:
            func_, args_, cols_ = _check_tuple(*tuple_)

            X_ = apply_func(
                X_,
                func_,
                args_,
                cols_,
                n_jobs=self.n_jobs,
                progress_bar=self.progress_bar,
            )

        return X_

    @staticmethod
    def apply_pandas(X_, func_, args_=None, cols_=None, **kwargs):
        """Apply a function on a pandas DataFrame.

        Parameters
        ----------
        X_ : pandas.DataFrame,
            Data on which transformations are applied.
        args_ : list or tuple
            List of arguments of the function to apply
        cols_ : list or tuple
            List of columns created by the transformation
        func_ : func
            Function to apply

        Returns
        -------
        pandas.DataFrame
        """
        if cols_ is None:
            X_ = X_.apply(func_, args=args_, axis=1)
        elif len(cols_) == 1:
            X_[cols_[0]] = X_.apply(func_, args=args_, axis=1)
        else:
            X_[cols_] = X_.apply(func_, args=args_, axis=1).apply(pd.Series)
        return X_

    @staticmethod
    def apply_pandas_multiprocessing(
        X_, func_, args_=None, cols_=None, n_jobs=1, progress_bar=False, **kwargs
    ):
        if cols_ is None:
            X_ = apply_by_multiprocessing(
                df=X_,
                func=func_,
                args=args_,
                axis=1,
                workers=n_jobs,
                progress_bar=progress_bar,
            )
        elif len(cols_) == 1:
            X_[cols_[0]] = apply_by_multiprocessing(
                df=X_,
                func=func_,
                args=args_,
                axis=1,
                workers=n_jobs,
                progress_bar=progress_bar,
            )
        else:
            X_[cols_] = apply_by_multiprocessing(
                df=X_,
                func=func_,
                args=args_,
                axis=1,
                workers=n_jobs,
                progress_bar=progress_bar,
            ).apply(pd.Series)
        return X_

    @staticmethod
    def apply_dict(X_, func_, args_=None, cols_=None, **kwargs):
        """Apply a function on a dictionary.

        Parameters
        ----------
        X_ : dict,
            Data on which transformations are applied.
        args_ : list or tuple
            List of arguments of the function to apply
        cols_ : list or tuple
            List of columns created by the transformation
        func_ : func
            Function to apply

        Returns
        -------
        dict
        """
        if not cols_:
            if not args_:
                X_ = func_(X_)
            else:
                X_ = func_(X_, *args_)
        elif len(cols_) == 1:
            if not args_:
                X_[cols_[0]] = func_(X_)
            else:
                X_[cols_[0]] = func_(X_, *args_)
        else:
            X_[cols_] = X_.apply(func_, args=args_, axis=1).apply(pd.Series)
            if not args_:
                X_.update(list(zip(cols_, func_(X_))))
            else:
                X_.update(list(zip(cols_, func_(X_, *args_))))

        return X_
