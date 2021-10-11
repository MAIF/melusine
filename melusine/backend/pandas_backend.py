from melusine.backend.base_backend import BaseTransformerBackend
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from tqdm import tqdm


class PandasBackend(BaseTransformerBackend):
    def __init__(self, progress_bar=False, workers=1):
        super().__init__()
        self.progress_bar = progress_bar
        self.workers = workers

    def apply_transform_(
        self, data, func, output_columns, input_columns=None, **kwargs
    ):
        # Multiprocessing
        if self.workers > 1:
            data = self.apply_transform_multiprocessing(
                data, func, output_columns, input_columns=input_columns, **kwargs
            )
        else:
            data = self.apply_transform_regular(
                data, func, output_columns, input_columns=input_columns, **kwargs
            )

        return data

    def apply_transform_regular(
        self, data, func, output_columns, input_columns=None, **kwargs
    ):

        if len(output_columns) > 1:
            expand = "expand"
            new_cols = list(output_columns)
        else:
            expand = None
            new_cols = output_columns[0]

        if input_columns and len(input_columns) == 1:
            input_column = input_columns[0]
            data[new_cols] = self.apply_joblib_series(
                s=data[input_column],
                func=func,
                expand=expand,
                progress_bar=self.progress_bar,
                **kwargs
            )

        else:
            data[new_cols] = self.apply_joblib_dataframe(
                df=data,
                func=func,
                expand=expand,
                progress_bar=self.progress_bar,
                **kwargs
            )

        return data

    def apply_transform_multiprocessing(
        self, data, func, output_columns, input_columns=None, **kwargs
    ):
        workers = min(self.workers, int(data.shape[0] // 2))
        workers = max(workers, 1)

        # Dataframe is too small to use multiprocessing
        if workers == 1:
            return self.apply_transform_regular(
                data, func, output_columns, input_columns=input_columns
            )

        if len(output_columns) > 1:
            expand = "expand"
            new_cols = output_columns[0]
        else:
            expand = None
            new_cols = output_columns

        if input_columns and len(input_columns) == 1:
            input_column = input_columns[0]
            chunks = Parallel(n_jobs=workers)(
                delayed(self.apply_joblib_series)(
                    s=d[input_column],
                    func=func,
                    expand=expand,
                    progress_bar=self.progress_bar,
                    **kwargs
                )
                for d in np.array_split(data, workers)
            )
        else:
            chunks = Parallel(n_jobs=workers)(
                delayed(self.apply_joblib_dataframe)(
                    df=d,
                    func=func,
                    expand=expand,
                    progress_bar=self.progress_bar,
                    **kwargs
                )
                for d in np.array_split(data, workers)
            )

        data[new_cols] = pd.concat(chunks)
        return data

    @staticmethod
    def apply_joblib_dataframe(df, func, expand=None, progress_bar=False, **kwargs):
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
    def apply_joblib_series(s, func, expand=None, progress_bar=False, **kwargs):
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
