"""
Backend to run transforms on pandas.DataFrame objects.

Implemented classes: [
    PandasBackend,
]
"""
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from melusine.backend.base_backend import BaseTransformerBackend


class PandasBackend(BaseTransformerBackend):
    """
    Backend class to operate on Pandas DataFrames.
    Inherits from the BaseTransformerBackend abstract class.
    Includes multiprocessing functionalities
    """

    def __init__(self, progress_bar: bool = False, workers: int = 1):
        """
        Parameters
        ----------
        progress_bar: bool
            If True, display progress bar
        workers: int
            Number of workers for multiprocessing
        """
        super().__init__()
        self.progress_bar = progress_bar
        self.workers = workers

    def apply_transform(
        self,
        data: pd.DataFrame,
        func: Callable,
        output_columns: Optional[List[str]] = None,
        input_columns: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Method to apply a transform on a Dataset using the Dict backend.

        Parameters
        ----------
        data: pd.DataFrame
            Data to be transformed
        func: Callable
            Transform function to apply to the input data
        output_columns: List[str]
            List of output columns
        input_columns: List[str]
            List of input columns
        kwargs

        Returns
        -------
        _: pd.DataFrame
            Transformed data
        """
        # Multiprocessing
        if self.workers > 1:
            data = self.apply_transform_multiprocessing(
                data, func, output_columns, input_columns=input_columns, **kwargs
            )
        else:
            data = self.apply_transform_regular(data, func, output_columns, input_columns=input_columns, **kwargs)

        return data

    def apply_transform_regular(
        self,
        data: pd.DataFrame,
        func: Callable,
        output_columns: Optional[List[str]] = None,
        input_columns: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Regular transform (no multiprocessing)

        Parameters
        ----------
        data: pd.DataFrame
            Data to be transformed
        func: Callable
            Transform function to apply to the input data
        output_columns: List[str]
            List of output columns
        input_columns: List[str]
            List of input columns
        kwargs

        Returns
        -------
        _: pd.DataFrame
            Transformed data
        """

        # Setup apply parameters
        expand, new_cols = self.setup_apply_parameters(output_columns)

        # Series apply
        if input_columns and len(input_columns) == 1:
            input_column = input_columns[0]

            result = self.apply_joblib_series(
                s=data[input_column],
                func=func,
                expand=expand,
                progress_bar=self.progress_bar,
                **kwargs,
            )

        # DataFrame apply
        else:
            result = self.apply_joblib_dataframe(
                df=data,
                func=func,
                expand=expand,
                progress_bar=self.progress_bar,
                **kwargs,
            )

        # Collect results
        if not new_cols:
            data = result
        else:
            data[new_cols] = result

        return data

    @staticmethod
    def setup_apply_parameters(
        output_columns: Optional[List[str]] = None,
    ) -> Tuple[Union[None, str], Union[None, str, List[str]]]:
        """
        Parameters
        ----------
        output_columns: List[str]
            List of output columns

        Returns
        -------
        expand: str
        new_cols: Union[None, str, List[str]]
        """
        if not output_columns:
            expand = None
            new_cols: Union[None, str, List[str]] = None
        elif len(output_columns) == 1:
            expand = None
            new_cols = output_columns[0]
        # Multiple output columns
        else:
            expand = "expand"
            new_cols = list(output_columns)

        return expand, new_cols

    def apply_transform_multiprocessing(
        self,
        data: pd.DataFrame,
        func: Callable,
        output_columns: Optional[List[str]] = None,
        input_columns: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Transform with multiprocessing

        Parameters
        ----------
        data: pd.DataFrame
            Data to be transformed
        func: Callable
            Transform function to apply to the input data
        output_columns: List[str]
            List of output columns
        input_columns: List[str]
            List of input columns
        kwargs

        Returns
        -------
        _: pd.DataFrame
            Transformed data
        """
        workers = min(self.workers, int(data.shape[0] // 2))
        workers = max(workers, 1)

        # Dataframe is too small to use multiprocessing
        if workers == 1:
            return self.apply_transform_regular(data, func, output_columns, input_columns=input_columns, **kwargs)

        expand, new_cols = self.setup_apply_parameters(output_columns)

        # Use Series.apply
        if input_columns and len(input_columns) == 1:
            input_column = input_columns[0]
            chunks = Parallel(n_jobs=workers)(
                delayed(self.apply_joblib_series)(
                    s=d[input_column],
                    func=func,
                    expand=expand,
                    progress_bar=self.progress_bar,
                    **kwargs,
                )
                for d in np.array_split(data, workers)
            )

        # Use DataFrame.apply
        else:
            chunks = Parallel(n_jobs=workers)(
                delayed(self.apply_joblib_dataframe)(
                    df=d,
                    func=func,
                    expand=expand,
                    progress_bar=self.progress_bar,
                    **kwargs,
                )
                for d in np.array_split(data, workers)
            )

        if not new_cols:
            data = pd.concat(chunks)
        else:
            data[new_cols] = pd.concat(chunks)

        return data

    @staticmethod
    def apply_joblib_dataframe(
        df: pd.DataFrame,
        func: Callable,
        expand: Optional[str] = None,
        progress_bar: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Need to create a function to pass to Joblib Parallel.
        This function can't be a lambda so we need to create a separate function.
        """
        if progress_bar:
            apply_func = "progress_apply"
            tqdm.pandas(desc=func.__name__)
        else:
            apply_func = "apply"

        result = getattr(df, apply_func)(func, axis=1, result_type=expand, **kwargs)

        return result

    @staticmethod
    def apply_joblib_series(
        s: pd.Series,
        func: Callable,
        expand: Optional[str] = None,
        progress_bar: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Need to create a function to pass to Joblib Parallel.
        This function can't be a lambda so we need to create a separate function.
        """
        if progress_bar:
            apply_func = "progress_apply"
            tqdm.pandas(desc=func.__name__)
        else:
            apply_func = "apply"

        result = getattr(s, apply_func)(func, **kwargs)
        if expand:
            result = result.apply(pd.Series)

        return result

    def add_fields(self, left: pd.DataFrame, right: pd.DataFrame, fields: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Method to add fields form the right object to the left object.

        Parameters
        ----------
        left: pd.DataFrame
            MelusineDataset object
        right: pd.DataFrame
            Melusine Dataset object
        fields: List[str]
            List of fields to be added

        Returns
        -------
        _: pd.DataFrame
            Left object with added fields
        """
        left[fields] = right[fields]

        return left

    def copy(self, data: pd.DataFrame, fields: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Method to make a copy of the dataset.

        Parameters
        ----------
        data: pd.DataFrame
            MelusineDataset object
        fields: List[str]
            List of fields to include in the copy (by default copy all fields)

        Returns
        -------
        _: pd.DataFrame
            Copy of original object
        """
        if not fields:
            fields = data.columns

        return data[fields].copy()

    def get_fields(self, data: pd.DataFrame) -> List[str]:
        """
        Method to get the list of fields available in the input dataset.

        Parameters
        ----------
        data: pd.DataFrame
            MelusineDataset object

        Returns
        -------
        _: List[str]
            List of dataset fields
        """
        return data.columns.to_list()

    def setup_debug_dict(self, data: pd.DataFrame, dict_name: str) -> pd.DataFrame:
        """
        Method to check if debug_mode is activated.

        Parameters
        ----------
        data: pd.DataFrame
            MelusineDataset object
        dict_name: str
            Name of the debug dict field to be added

        Returns
        -------
        _: pd.DataFrame
            MelusineDataset object
        """
        data[dict_name] = [{} for _ in range(len(data))]

        return data
