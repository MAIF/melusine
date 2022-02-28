import numpy
import pandas
from sklearn.base import BaseEstimator, TransformerMixin


class VerbitimTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, func, kw_args, in_col, out_col):
        """
        Instanciates a VerbitimTransformer.
        This object wraps functions that operates on a pandas dataframe in sklearn Transforme compatible inside sklearn
        Pipeline. Enables user to provide input and output columns.
        For more information on transformers and pipelines. Please refer to the scikit-learn documentation
        Parameters
        ----------
        func : callable
        Function to wrap
        kw_args : dict
        Keywords arguments to pass along with func
        in_col : str
        Input column
        out_col : str
        Output column
        """
        def __check_function_type(fn: callable) -> callable:
            if not callable(fn):
                raise TypeError("First item of the tuple (func, args, in_col, out_col) must be a \
                    function-like object not a {} object".format(type(fn)))
            else:
                return fn

        def __check_kw_args_type(args: dict) -> dict:
            if args is None or args == {}:
                return None
            elif not isinstance(args, dict):
                raise TypeError("Second item of the tuple (func, args, in_col, out_col) must be dict-like \
                    object not a {} object".format(type(args)))
            else:
                return args

        def __check_colname_type(col: str) -> str:
            if col is None or col == []:
                return None
            elif not isinstance(col, str):
                raise TypeError("Third item of the tuple (func, args, in_col, out_col) must be str \
                    object not a {} object".format(type(col)))
            else:
                return col

        self.func = __check_function_type(func)
        self.kw_args = __check_kw_args_type(kw_args)
        self.in_col = __check_colname_type(in_col)
        self.out_col = __check_colname_type(out_col)

    def fit(self, X, y=None):
        """Unused method. Defined only for compatibility with scikit-learn API."""
        return self

    def transform(self, df: pandas.DataFrame) -> pandas.DataFrame:
        df[self.out_col] = self.func(df[self.in_col], **(self.kw_args if self.kw_args else {}))
        return df
