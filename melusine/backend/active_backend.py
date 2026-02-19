"""Melusine transformation can operate on different data structures such as dict or pandas.DataFrame.
Different transformation backends are used to process different data structures.
The ActiveBackend class stores an instance of the activated backend.

Implemented classes: [
    ActiveBackend,
]
"""

import logging
from collections.abc import Callable
from itertools import chain
from typing import Any

from melusine.backend.base_backend import BaseTransformerBackend
from melusine.backend.dict_backend import DictBackend
from melusine.backend.pandas_backend import PandasBackend

logger = logging.getLogger(__name__)


class ActiveBackend(BaseTransformerBackend):
    """Class storing the active backend used by Melusine."""

    PANDAS_BACKEND: str = "pandas"
    DICT_BACKEND: str = "dict"

    def __init__(self) -> None:
        """Init"""
        super().__init__()
        self.backend_list: list[BaseTransformerBackend] = []

    @property
    def supported_types(self) -> tuple:
        """Supported types for the active backend."""
        supported_types_tuples = [_backend.supported_types for _backend in self.backend_list]

        return tuple(set(chain(*supported_types_tuples)))

    def add(
        self,
        new_backend: BaseTransformerBackend | str,
    ) -> None:
        """Method to add a backend to the active backend list.

        Parameters
        ----------
        new_backend: BaseTransformerBackend | str
            New backend to be used

        """
        if isinstance(new_backend, BaseTransformerBackend):
            self.backend_list.append(new_backend)

        elif new_backend == self.PANDAS_BACKEND:
            self.backend_list.append(PandasBackend())

        elif new_backend == self.DICT_BACKEND:
            self.backend_list.append(DictBackend())

        else:
            raise ValueError(f"Backend {new_backend} is not supported")

        logger.info(f"Using backends '{self.backend_list}' for Data transformations")

    def reset(
        self, new_backend: BaseTransformerBackend | str | None = None, keep_default_backends: bool = True
    ) -> None:
        """Method to reset active backend list.

        Parameters
        ----------
        new_backend: BaseTransformerBackend | str
            New backend to be used
        keep_default_backends: If True, keep the default backends on top of the new one.

        """
        if keep_default_backends:
            self.backend_list = [DictBackend(), PandasBackend()]
        else:
            self.backend_list = []

        if new_backend:
            self.add(new_backend)

    def apply_transform(
        self,
        data: Any,
        func: Callable,
        output_columns: list[str] | None = None,
        input_columns: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Method to apply a transform on a Dataset using current backend.

        Parameters
        ----------
        data: Dataset
            Data to be transformed
        func: Callable
            Transform function to apply to the input data
        output_columns: Sequence[str]
            List of output columns
        input_columns: Sequence[str]
            List of input columns
        kwargs

        Returns
        -------
        _: Dataset
            Transformed data

        """
        _backend = self.select_backend(data=data)
        return _backend.apply_transform(
            data=data,
            func=func,
            output_columns=output_columns,
            input_columns=input_columns,
            **kwargs,
        )

    def select_backend(self, data: Any):
        """Automatically select appropriate backend based on data type."""
        for _backend in self.backend_list[::-1]:
            if isinstance(data, _backend.supported_types):
                return _backend

        raise ValueError(
            f"Could not find an appropriate backend fo data of type {type(data)}\n"
            f"Backends available are: {self.backend_list}\n"
            "To add an extra backend, use backend.add_backend(my_backend)"
        )

    def copy(self, data: Any, fields: list[str] | None = None) -> Any:
        """Method to make a copy of the input dataset.

        Parameters
        ----------
        data: Dataset
            MelusineDataset object
        fields: List[str]
            List of fields to include in the copy (by default copy all fields)

        Returns
        -------
        _: Dataset
            Copy of original object

        """
        _backend = self.select_backend(data=data)
        return _backend.copy(data, fields=fields)

    def get_fields(self, data: Any) -> list[str]:
        """Method to get the list of fields available in the input dataset.

        Parameters
        ----------
        data: Dataset
            MelusineDataset object

        Returns
        -------
        _: List[str]
            List of dataset fields

        """
        _backend = self.select_backend(data=data)
        return _backend.get_fields(data=data)

    def add_fields(self, left: Any, right: Any, fields: list[str] | None = None) -> Any:
        """Method to add fields from the right object to the left object

        Parameters
        ----------
        left: Dataset
            MelusineDataset object
        right: Dataset
            Melusine Dataset object
        fields: List[str]
            List of fields to be added

        Returns
        -------
        _: Dataset
            Left object with added fields

        """
        _backend = self.select_backend(data=left)
        return _backend.add_fields(left=left, right=right, fields=fields)

    def setup_debug_dict(self, data: Any, dict_name: str) -> Any:
        """Method to check if debug_mode is activated

        Parameters
        ----------
        data: Dataset
            MelusineDataset object
        dict_name: str
            Name of the debug dict field to be added

        Returns
        -------
        _: Dataset
            MelusineDataset object

        """
        _backend = self.select_backend(data=data)
        return _backend.setup_debug_dict(data=data, dict_name=dict_name)


# Instantiate the default backend
backend = ActiveBackend()
backend.reset()
