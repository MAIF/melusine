import logging
import os
import re

import pytest

from melusine import config
from melusine._config import MelusineConfig, MelusineConfigError


def test_load_default_conf(caplog):
    with caplog.at_level(logging.WARNING):
        test_conf = MelusineConfig()
        test_conf.reset()

        assert test_conf
        assert MelusineConfig.LOG_MESSAGE_DEFAULT_CONFIG in caplog.text


def test_load_conf_from_env_variable(caplog):
    try:
        with caplog.at_level(logging.INFO):
            test_conf = MelusineConfig()
            os.environ[MelusineConfig.ENV_MELUSINE_CONFIG_DIR] = test_conf.DEFAULT_CONFIG_PATH
            test_conf.reset()

            expected_config_path_log = MelusineConfig.LOG_MESSAGE_CONFIG_PATH.format(
                config_path=test_conf.DEFAULT_CONFIG_PATH
            )

            assert test_conf
            assert MelusineConfig.LOG_MESSAGE_CONFIG_FROM_ENV_VARIABLE in caplog.text
            assert expected_config_path_log in caplog.text
            assert MelusineConfig.LOG_MESSAGE_DEFAULT_CONFIG not in caplog.text

    finally:
        del os.environ[MelusineConfig.ENV_MELUSINE_CONFIG_DIR]


def test_load_conf_from_config_path(caplog):
    with caplog.at_level(logging.INFO):
        test_conf = MelusineConfig()
        test_conf.reset(config_path=test_conf.DEFAULT_CONFIG_PATH)

        expected_config_path_log = MelusineConfig.LOG_MESSAGE_CONFIG_PATH.format(
            config_path=test_conf.DEFAULT_CONFIG_PATH
        )

        assert test_conf
        assert expected_config_path_log in caplog.text
        assert MelusineConfig.LOG_MESSAGE_CONFIG_FROM_ENV_VARIABLE not in caplog.text
        assert MelusineConfig.LOG_MESSAGE_DEFAULT_CONFIG not in caplog.text


def test_load_conf_from_config_dict(caplog):
    with caplog.at_level(logging.INFO):
        test_conf = MelusineConfig()
        test_conf.reset(config_dict={"my_key": "hello"})

        assert test_conf["my_key"] == "hello"


def test_config_modif_error():
    test_conf = MelusineConfig()
    test_conf.reset(config_dict={"my_key": "hello"})

    with pytest.raises(MelusineConfigError, match=re.escape(MelusineConfigError.CONST_CONFIG_ERROR_MESSAGE)):
        test_conf["new_key"] = "hey"

    with pytest.raises(MelusineConfigError):
        test_conf.pop()

    with pytest.raises(MelusineConfigError):
        test_conf.popitem()


def test_shared_variable():
    # Shared variable TEST_VAR specified in conf/shared.yaml
    # Conf test_shared_variable specified in global.yaml
    assert config["global"]["test_shared_variable"] == "test"


def test_export_config(tmp_path):
    file_list = config.export_default_config(path=str(tmp_path))
    assert file_list
    for file in file_list:
        assert file.endswith(".yaml")
