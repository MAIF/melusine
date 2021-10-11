import os
import logging

from melusine.backend.base_backend import BaseTransformerBackend
from melusine.backend.dict_backend import DictBackend

logger = logging.getLogger(__name__)


class ActiveBackend:
    PANDAS_BACKEND = "pandas"
    DICT_BACKEND = "dict"

    def __init__(self, new_backend, **kwargs):
        super().__init__()
        self._backend = None
        self._switch_backend(new_backend, **kwargs)

    def _switch_backend(self, new_backend, **kwargs):

        if new_backend == self.PANDAS_BACKEND:
            # Importing in local scope to prevent hard dependencies
            from melusine.backend.pandas_backend import PandasBackend

            self._backend = PandasBackend(**kwargs)

        elif new_backend == self.DICT_BACKEND:
            self._backend = DictBackend(**kwargs)

        elif isinstance(new_backend, BaseTransformerBackend):
            self._backend = new_backend

        else:
            raise AttributeError(f"Backend {new_backend} is not supported")

        print(f"Using {new_backend} backend for Data transformations")
        logger.info(f"Using backend '{new_backend}' for Data transformations")

    def apply_transform(
        self, data, func, output_columns, input_columns=None, *args, **kwargs
    ):
        return self._backend.apply_transform_(
            data=data,
            func=func,
            output_columns=output_columns,
            input_columns=input_columns,
            *args,
            **kwargs,
        )


def switch_backend(new_backend, **kwargs):
    global backend

    backend._switch_backend(new_backend, **kwargs)


backend = ActiveBackend(
    new_backend=os.getenv("MELUSINE_BACKEND", ActiveBackend.PANDAS_BACKEND)
)
