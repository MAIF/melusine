from melusine.backend.melusine_backend import BaseTransformerBackend


class PandasBackend(BaseTransformerBackend):
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
