from abc import ABC, abstractmethod


class BaseTransformerBackend(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def apply_transform_(
        self, data, func, output_columns, input_columns=None, **kwargs
    ):
        return NotImplementedError
