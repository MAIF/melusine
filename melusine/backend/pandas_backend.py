from melusine.backend.base_backend import BaseTransformerBackend


class PandasBackend(BaseTransformerBackend):
    def __init__(self, progress_bar=False):
        super().__init__()
        self.progress_bar = progress_bar
        if self.progress_bar:
            self.apply_transform_ = self.apply_transform_progress_bar

    @staticmethod
    def apply_transform_(data, func, output_columns, input_columns=None):
        import pandas as pd

        if not input_columns:
            input_columns = data.columns.tolist()

        if len(input_columns) == 1:
            input_column = input_columns[0]
            if len(output_columns) == 1:
                output_column = output_columns[0]
                data[output_column] = data[input_column].apply(func)
            else:
                data[output_columns] = data[input_column].apply(func).apply(pd.Series)
        else:
            data[output_columns] = data[input_columns].apply(
                func, axis=1, expand_results=True
            )
        return data

    @staticmethod
    def apply_transform_progress_bar(data, func, output_columns, input_columns=None):
        import pandas as pd
        from tqdm import tqdm

        tqdm.pandas(desc=func.__name__)
        if not input_columns:
            input_columns = data.columns.tolist()

        if len(input_columns) == 1:
            input_column = input_columns[0]
            if len(output_columns) == 1:
                output_column = output_columns[0]
                data[output_column] = data[input_column].progress_apply(func)
            else:
                data[output_columns] = (
                    data[input_column].apply(func).progress_apply(pd.Series)
                )
        else:
            data[output_columns] = data[input_columns].progress_apply(
                func, axis=1, expand_results=True
            )
        return data
