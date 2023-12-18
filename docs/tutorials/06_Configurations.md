# Configurations

Melusine components can be instantiated using parameters defined in configurations.
The `from_config` method accepts a `config_dict` argument
```Python
--8<--
docs/docs_src/Configurations/tutorial001.py:from_config_dict
--8<--
```

or a `config_key` argument.
```Python
--8<--
docs/docs_src/Configurations/tutorial001.py:from_config
--8<--
```
When `demo_pipeline` is given as argument, parameters are read from the `melusine.config` object at key `demo_pipeline`. 

## Access configurations

The melusine configurations can be accessed with the `config` object.
```Python
--8<--
docs/docs_src/Configurations/tutorial001.py:print_config
--8<--
```

The configuration of the `demo_pipeline` can then be easily inspected.

```Python
{
  'steps': [
    {'class_name': 'Cleaner', 'config_key': 'body_cleaner', 'module': 'melusine.processors'},
    {'class_name': 'Cleaner', 'config_key': 'header_cleaner', 'module': 'melusine.processors'},
    {'class_name': 'Segmenter', 'config_key': 'segmenter', 'module': 'melusine.processors'},
    {'class_name': 'ContentTagger', 'config_key': 'content_tagger', 'module': 'melusine.processors'},
    {'class_name': 'TextExtractor', 'config_key': 'text_extractor', 'module': 'melusine.processors'},
    {'class_name': 'Normalizer', 'config_key': 'demo_normalizer', 'module': 'melusine.processors'},
    {'class_name': 'EmergencyDetector', 'config_key': 'emergency_detector', 'module': 'melusine.detectors'}
  ]
}
```

## Modify configurations
The simplest way to modify configurations is to create a new directory directly.
```Python
--8<--
docs/docs_src/Configurations/tutorial001.py:modify_conf_with_dict
--8<--
```

To deliver code in a production environment, using configuration files should be preferred to
modifying the configurations on the fly.  
Melusine lets you specify the path to a folder containing *yaml* files and loads them (the `OmegaConf` package is used behind the scene).
```Python
--8<--
docs/docs_src/Configurations/tutorial001.py:modify_conf_with_path
--8<--
```

When the `MELUSINE_CONFIG_DIR` environment variable is set, Melusine loads directly the configurations files located at
the path specified by the environment variable.
```Python
--8<--
docs/docs_src/Configurations/tutorial001.py:modify_conf_with_env
--8<--
```

!!! tip
    If the `MELUSINE_CONFIG_DIR` is set before melusine is imported (e.g., before starting the program), you don't need to call `config.reset()`. 

## Export configurations

Creating your configuration folder from scratch would be cumbersome.
It is advised to export the default configurations and then modify just the files you need.

```Python
--8<--
docs/docs_src/Configurations/tutorial001.py:export_config
--8<--
```

!!! tip
    The `export_default_config` returns a list of path to all the files created. 
