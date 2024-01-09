"""
Contain IO classes implementation.

Contained classes: [IoMixin]
"""
from __future__ import annotations

import logging
from typing import Any, TypeVar

from melusine import config

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="IoMixin")


class InitError(Exception):
    """
    Error raised when object instantiation fails.
    """


class IoMixin:
    """
    Defines generic load methods.
    """

    def __init__(self, **kwargs: Any):
        """Initialize attribute."""
        self.json_exclude_list: list[str] = ["_func", "json_exclude_list"]

    @classmethod
    def from_config(
        cls: type[T],
        config_key: str | None = None,
        config_dict: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> T:
        """
        Instantiate a class from a config key or a config dict.

        Parameters
        ----------
        config_key: str
            Configuration key.
        config_dict: dict[str, Any]
            Dictionary of config.
        kwargs: Any

        Returns
        -------
        _: T
            Instantiated objet.
        """
        # Load from Melusine config
        if config_dict is None:
            if config_key is None:
                raise ValueError("You should specify one and only one of 'config_key' and 'config_value'")
            else:
                config_dict = config[config_key]
        else:
            if config_key is not None:
                raise ValueError("You should specify one and only one of 'config_key' and 'config_value'")

        # Update with keyword arguments
        config_dict.update(**kwargs)

        return cls.from_dict(**config_dict)

    @classmethod
    def from_dict(cls: type[T], **params_dict: dict[str, Any]) -> T:
        """
        Method to instantiate a class based a dict object.

        Parameters
        ----------
        params_dict: dict[str, Any]
            Parameters dict.

        Returns
        -------
        _: T
            Instantiated objet.
        """
        # Exclude parameters starting with an underscore
        init_params = {key: value for key, value in params_dict.items() if not key.startswith("_")}

        try:
            instance = cls(**init_params)
            return instance
        except Exception as error:
            raise InitError(f"Failed to instantiate {cls.__name__} with attributes {init_params}.").with_traceback(
                error.__traceback__
            )
