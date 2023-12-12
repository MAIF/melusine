import pytest

from melusine import config
from melusine.io import IoMixin
from melusine.io._classes import InitError


class FakeClass(IoMixin):
    def __init__(self, test_arg):
        super().__init__()
        self.test_arg = test_arg


def test_from_config_dict():
    config_dict = {"test_arg": "hey"}
    instance = FakeClass.from_config(config_dict=config_dict)

    assert instance.test_arg == "hey"


@pytest.mark.usefixtures("use_dict_backend", "reset_melusine_config")
def test_from_config_key():
    my_dict = {"test_arg": "hey"}
    test_conf_dict = config.dict()
    test_conf_dict["testclass_conf"] = my_dict
    config.reset(config_dict=test_conf_dict)

    instance = FakeClass.from_config(config_key="testclass_conf")

    assert instance.test_arg == "hey"


def test_from_config_dict_error():
    unknown_arg = "unknown_arg"
    config_dict = {"test_arg": "hey", unknown_arg: 42}

    with pytest.raises(InitError, match=f"{FakeClass.__name__}.*{unknown_arg}"):
        _ = FakeClass.from_config(config_dict=config_dict)


def test_from_config_dict_and_config_key_error():
    config_dict = {"test_arg": "hey"}

    with pytest.raises(ValueError):
        _ = FakeClass.from_config(config_dict=config_dict, config_key="blabla")


def test_from_config_dict_and_config_key_none_error():
    with pytest.raises(ValueError):
        _ = FakeClass.from_config()
