def test_tutorial001(add_docs_to_pythonpath):
    from docs_src.Configurations.tutorial001 import (
        from_config,
        from_config_dict,
        modify_conf_with_dict,
        print_config,
    )

    _ = from_config()
    _ = from_config_dict()
    _ = print_config()
    _ = modify_conf_with_dict()
