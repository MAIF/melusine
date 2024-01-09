"""
Melusine transformation can operate on different data structures such as dict or pandas.DataFrame.
Different transformation backends are used to process different data structures.
The BaseTransformerBackend class defines the interface for transformation backend classes.

Implemented classes: [
    BaseTransformerBackend,
]
"""
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional


class BaseTransformerBackend(ABC):
    """
    Abstract base class defining how to implement a Melusine Backend.
    Each backend applies transform operations on a specific type of data.
    Ex: Pandas DataFrames, Dict, Spark objects, etc
    """

    DEBUG_FLAG = "debug"

    @abstractmethod
    def apply_transform(
        self,
        data: Any,
        func: Callable,
        output_columns: Optional[List[str]] = None,
        input_columns: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Method to apply a transform on a Dataset using the current backend.

        Parameters
        ----------
        data: Any
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
        _: Any
            Transformed data
        """

    @abstractmethod
    def add_fields(self, left: Any, right: Any, fields: Optional[List[str]] = None) -> Any:
        """
        Method to add fields form the right object to the left object.

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

    @abstractmethod
    def copy(self, data: Any, fields: Optional[List[str]] = None) -> Any:
        """
        Method to make a copy of the dataset.

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

    @abstractmethod
    def get_fields(self, data: Any) -> List[str]:
        """
        Method to get the list of fields available in the input dataset.

        Parameters
        ----------
        data: Dataset
            MelusineDataset object

        Returns
        -------
        _: List[str]
            List of dataset fields
        """

    def check_debug_flag(self, data: Any) -> bool:
        """
        Method to check if debug_mode is activated.

        Parameters
        ----------
        data: Dataset
            MelusineDataset object

        Returns
        -------
        _: bool
            True if debug mode is activated
        """
        if hasattr(data, self.DEBUG_FLAG):
            debug_mode = getattr(data, self.DEBUG_FLAG)
        elif isinstance(data, dict):
            debug_mode = data.get(self.DEBUG_FLAG, False)
        else:  # pragma: no cover
            debug_mode = False

        return debug_mode

    @abstractmethod
    def setup_debug_dict(self, data: Any, dict_name: str) -> Any:
        """
        Method to check if debug_mode is activated.

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
