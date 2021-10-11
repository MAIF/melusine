from melusine.backend.base_backend import BaseTransformerBackend


class DictBackend(BaseTransformerBackend):
    def __init__(self, **kwargs):
        super().__init__()
        pass

    def apply_transform_(
        self, data, func, output_columns, input_columns=None, **kwargs
    ):

        if input_columns and len(input_columns) == 1:
            input_column = input_columns[0]
            if len(output_columns) == 1:
                output_column = output_columns[0]
                data[output_column] = func(data[input_column])
            else:
                result = func(data[input_column])
                data.update(dict(zip(output_columns, result)))
        else:
            if len(output_columns) == 1:
                output_column = output_columns[0]
                data[output_column] = func(data)
            else:
                result = func(data)
                data.update(dict(zip(output_columns, result)))

        return data
