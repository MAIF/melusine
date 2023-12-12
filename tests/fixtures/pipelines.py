import pytest

from melusine.pipeline import MelusinePipeline


@pytest.fixture
def conf_pipeline_basic():
    return {
        "test_pipeline": {
            "steps": [
                {
                    "class_name": "Normalizer",
                    "module": "melusine.processor",
                    "config_key": "test_normalizer",
                },
                {
                    "class_name": "RegexTokenizer",
                    "module": "melusine.processor",
                    "config_key": "test_tokenizer",
                },
            ]
        }
    }


@pytest.fixture
def pipeline_default():
    # Load json config
    conf = MelusinePipeline.get_config_from_key(config_key="my_pipeline")

    # Prevent model loading
    # (Set all model_name parameters to None)
    for step in conf["steps"]:
        step["name"] = step.pop("config_key", None)
        params = step["parameters"]

        if "model_name" in params:
            params["model_name"] = None

    # Create pipeline from a json config file (using config key "my_pipeline")
    return MelusinePipeline.from_config(config_dict=conf, verbose=True)
