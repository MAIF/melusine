import os
from melusine import config
from melusine.config.config import switch_config, MelusineConfig
from tempfile import TemporaryDirectory
import yaml


def test_config():

    # Config is not empty
    assert (len(config)) > 0

    # Config has keys
    assert config.keys()

    # Config has values
    assert config.keys()

    # Config has items
    assert config.items()

    # Switch config
    new_config = config.copy()
    test_key = list(new_config.keys())[0]
    test_value = "TEST"
    new_config[test_key] = test_value
    switch_config(new_config)

    assert config[test_key] == test_value

    # Load custom config
    custom_test_key = "custom_test_key"
    data = {custom_test_key: test_value}

    with TemporaryDirectory() as tmpdir:
        tmp_yaml_file = os.path.join(tmpdir, "test.yml")
        with open(tmp_yaml_file, "w") as f:
            yaml.dump(data, f)

        os.environ["MELUSINE_CONFIG_DIR"] = os.path.join(os.getcwd(), tmpdir)
        custom_config = MelusineConfig()
        assert custom_config[custom_test_key] == test_value
