from abc import ABC, abstractmethod


class BaseTransformerBackend(ABC):
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def apply_transform_(data, func, output_columns, input_columns=None):
        return NotImplementedError
