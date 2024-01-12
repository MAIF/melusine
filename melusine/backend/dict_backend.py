"""
Backend to run transforms on dict objects.

Implemented classes: [
    DictBackend,
]
"""

from typing import Any, Callable, Dict, List, Optional

from melusine.backend.base_backend import BaseTransformerBackend


class DictBackend(BaseTransformerBackend):
    """
    Backend class to operate on dict objects.
    Inherits from the BaseTransformerBackend abstract class.
    """

    def apply_transform(
        self,
        data: Dict[str, Any],
        func: Callable,
        output_columns: Optional[List[str]] = None,
        input_columns: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Method to apply a transform on a Dataset using the Dict backend.

        Parameters
        ----------
        data: Dict[str: Any]
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
        _: Dict[str: Any]
            Transformed data
        """
        if input_columns and len(input_columns) == 1:
            input_column = input_columns[0]

            # Modify the entire dict
            if not output_columns:
                raise ValueError("DictBackend does not support single input + None output situation.")

            # Create a single new field
            elif len(output_columns) == 1:
                output_column = output_columns[0]
                data[output_column] = func(data[input_column], **kwargs)

            # Create multiple new fields
            else:
                result = func(data[input_column], **kwargs)
                data.update(dict(zip(output_columns, result)))

        # Use DataFrame.apply
        else:
            # Modify the entire dict
            if not output_columns:
                data = func(data, **kwargs)

            # Create a single new field
            elif len(output_columns) == 1:
                output_column = output_columns[0]
                data[output_column] = func(data, **kwargs)

            # Create multiple new fields
            else:
                result = func(data, **kwargs)
                data.update(dict(zip(output_columns, result)))

        return data

    def add_fields(
        self, left: Dict[str, Any], right: Dict[str, Any], fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Method to add fields form the right object to the left object.

        Parameters
        ----------
        left: Dict[str, Any]
            MelusineDataset object
        right: Dict[str, Any]
            Melusine Dataset object
        fields: List[str]
            List of fields to be added

        Returns
        -------
        _: Dict[str, Any]
            Left object with added fields
        """
        if not fields:
            fields = list(right.keys())

        for field in fields:
            left[field] = right[field]

        return left

    def copy(self, data: Dict[str, Any], fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Method to make a copy of the dataset.

        Parameters
        ----------
        data: Dict[str, Any]
            MelusineDataset object
        fields: List[str]
            List of fields to include in the copy (by default copy all fields)

        Returns
        -------
        _: Dict[str, Any]
            Copy of original object
        """
        new_dict = dict()

        if fields is None:
            fields = list(data.keys())

        for field in fields:
            new_dict[field] = data[field]

        return new_dict

    def get_fields(self, data: Dict[str, Any]) -> List[str]:
        """
        Method to get the list of fields available in the input dataset.

        Parameters
        ----------
        data: Dict[str, Any]
            MelusineDataset object

        Returns
        -------
        _: List[str]
            List of dataset fields
        """
        return list(data.keys())

    def setup_debug_dict(self, data: Dict[str, Any], dict_name: str) -> Dict[str, Any]:
        """
        Method to check if debug_mode is activated.

        Parameters
        ----------
        data: Dict[str, Any]
            MelusineDataset object
        dict_name: str
            Name of the debug dict field to be added

        Returns
        -------
        _: Dict[str, Any]
            MelusineDataset object
        """
        data[dict_name] = {}

        return data
