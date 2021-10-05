import os
import logging
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


class ActiveBackend:
    PANDAS_BACKEND = "pandas"
    DICT_BACKEND = "dict"

    def __init__(self, new_backend):
        super().__init__()
        self._backend = None
        self.switch_backend(new_backend)

    def switch_backend(self, new_backend):

        if new_backend == self.PANDAS_BACKEND:
            from melusine.backend.pandas_backend import PandasBackend

            self._backend = PandasBackend()

        elif new_backend == self.DICT_BACKEND:
            self._backend = DictBackend()

        elif isinstance(new_backend, BaseTransformerBackend):
            self._backend = new_backend

        else:
            raise AttributeError(f"Backend {new_backend} is not supported")

        print(f"Using {new_backend} backend for Data transformations")
        logger.info(f"Using backend '{new_backend}' for Data transformations")

    def apply_transform(self, data, func, output_columns, input_columns=None):
        return self._backend.apply_transform_(
            data=data,
            func=func,
            output_columns=output_columns,
            input_columns=input_columns,
        )


class BaseTransformerBackend(ABC):
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def apply_transform_(data, func, output_columns, input_columns=None):
        return NotImplementedError


class DictBackend(BaseTransformerBackend):
    @staticmethod
    def apply_transform_(data, func, output_columns, input_columns=None):

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


def use(new_backend):
    global backend

    backend.switch_backend(new_backend)


backend = ActiveBackend(
    new_backend=os.getenv("MELUSINE_BACKEND", ActiveBackend.PANDAS_BACKEND)
)
