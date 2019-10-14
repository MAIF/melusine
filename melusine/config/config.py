import json
import os.path as op
import logging
from configparser import ConfigParser
import pandas as pd
import unidecode

class ConfigJsonReader():
    """Class to initialize a config.json file that contains your own
    parameters used by the package.

    In fact, the module `prepare_email` contains a set of functions which use "parameters"
    pre-defined in the conf.json.

    The json file contains parameters, such as:
        - general or specific regular expressions
        - footers, greetings or other mail keywords/tags

    Attributes
    ----------
    path_ini_file_ : str,
        Path to the path.ini file (file used only to check,
        set or get path of conf.json).

    path_default_conf_json_ : str,
        Path to the default conf.json (define by the contributors).

    Examples
    --------
    >>> conf = ConfigJsonReader()
    >>> conf.path_ini_file_   # will return path to the created .ini file

    >>> # I defined my own path otherwise it uses a default conf.json
    >>> conf.set_config_path(file_path='path/to/my/conf.json')
    >>> conf_dict = conf.get_config_file()
    >>> print(conf_dict)  # will print the json.

    """

    def __init__(self):
        config_directory = op.dirname(op.abspath(__file__))
        self.path_ini_file_ = op.join(config_directory, 'path.ini')
        self.path_default_conf_json_ = op.join(config_directory, 'conf.json')
        self.path_default_names_csv_ = op.join(config_directory, 'names.csv')

        if not op.exists(self.path_ini_file_):
            logging.info("Create an path.ini file to configure your own config.json")
            ini_file = open(self.path_ini_file_, 'w')
            conf = ConfigParser()
            conf.add_section('PATH')
            conf.set('PATH', 'template_config', self.path_default_conf_json_)
            conf.set('PATH', 'default_name_file', self.path_default_names_csv_)
            conf.write(ini_file)
            ini_file.close()

        self.config = ConfigParser()
        self.config.read(self.path_ini_file_)

        pass

    def set_config_path(self, file_path=None):
        """Set a path for your own `config.json`.

        Parameters
        ----------
        file_path: str, optional
            Path to the json file. If set to None (default), it will use the default
            json located in the built-in package `melusine/config/conf.json`.
        """
        if file_path is not None:
            # if file_path is specified, it writes new path in the .ini file.
            self.config['PATH']['template_config'] = file_path
            with open(self.path_ini_file_, 'w') as configfile:
                self.config.write(configfile)
        pass

    def get_config_file(self):
        """Load a config json file from the given path.
           Load the list of names from the names.csv file.
        """
        path = self.config['PATH']['template_config']
        if path == self.path_default_conf_json_:
            config_file = self.load_config_file(path=None)
        else:
            config_file = self.load_config_file(path=path)
        name_file_path = self.config['PATH']['default_name_file']

        if name_file_path == self.path_default_names_csv_:
            name_list = self.load_name_file(path=None)
        else:
            name_list = self.load_name_file(path=name_file_path)

        config_file['words_list']['names'] = name_list

        return config_file

    def reset_config_path(self):
        self.config['PATH']['template_config'] = self.path_default_conf_json_
        with open(self.path_ini_file_, 'w') as configfile:
            self.config.write(configfile)

        pass

    def load_config_file(self, path=None):
        """Load Json."""
        # by default it takes native the config.json
        if path is None:
            path = self.path_default_conf_json_

        with open(file=path, mode='r', encoding='utf-8') as file:
            config_file = json.load(file)

        return config_file

        logging.info("Load config from path: {}.".format(path))

    def set_name_file_path(self, file_path=None):
        """Set a path for your own `names.csv`.

        Parameters
        ----------
        file_path: str, optional
            Path to the csv file. If set to None (default), it will use the default
            csv located in the built-in package `melusine/config/names.csv`.
        """
        if file_path is not None:
            # if file_path is specified, it writes new path in the .ini file.
            self.config['PATH']['default_name_file'] = file_path
            with open(self.path_ini_file_, 'w') as configfile:
                self.config.write(configfile)
        pass

    def reset_name_file_path(self):
        self.config['PATH']['default_name_file'] = self.path_default_names_csv_
        with open(self.path_ini_file_, 'w') as configfile:
            self.config.write(configfile)
        pass

    def load_name_file(self, path):
        """Load csv."""
        # by default it takes native the names.csv
        if path is None:
            path = self.path_default_names_csv_

        try:
            df_names = pd.read_csv(path, encoding='utf-8', sep=";")
            name_list = df_names['Name'].values
            name_list = [unidecode.unidecode(p).lower() for p in name_list]
        except FileNotFoundError:
            name_list = []

        return name_list
