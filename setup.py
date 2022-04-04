#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
import glob
from setuptools import setup

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = [    
    "tensorflow>=2.8.0",
    "pandas>=1.3.0",
    "scikit-learn>=0.23",
    "gensim>=4.1.2",
    "tqdm>=4.34",
    "unidecode>=1.0",
    "flashtext>=2.7",
    "h5py>=3.0",
    "joblib>=1.0",
    "PyYAML>=4.2",
]

# Optional dependencies
exchange_requirements = ["exchangelib>=4.2.0"]
transformers_requirements = ["transformers==3.4.0"]
viz_requirements = ["plotly", "streamlit>=0.57.3"]
lemmatizer_requirements = ["spacy>=3.0.0,<=3.0.4", "spacy-lefff==0.4.0"]
stemmer_requirements = ["nltk>=3.6.7"]
emoji_requirements = ["emoji>=1.6.3"]


# Test dependencies
setup_requirements = ["pytest-runner"]
test_requirements = transformers_requirements + ["pytest"]


# Ex: Install all dependencies with ``pip install melusine[all]`
extras_require = {
    "exchange": exchange_requirements,
    "transformers": transformers_requirements,
    "viz": viz_requirements,
    "lemmatizer": lemmatizer_requirements,
    "stemmer": stemmer_requirements,
    "emoji": emoji_requirements,
}
all_requirements = list(set([y for x in extras_require.values() for y in x]))
extras_require["all"] = all_requirements

# Conf files
conf_json_files = glob.glob("melusine/config/**/*.json")


setup(
    author="Sacha Samama, Tom Stringer, Antoine Simoulin, Benoit Lebreton, Tiphaine Fabre, Hugo Perrier",
    author_email="tiphaine.fabre@maif.fr",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description=(
        """Melusine is a high-level package for french emails preprocessing, """
        """classification and feature extraction, written in Python."""
    ),
    entry_points={},
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="melusine",
    name="melusine",
    package_dir={
        "melusine": "melusine",
        "melusine.config": "melusine/config",
        "melusine.utils": "melusine/utils",
        "melusine.nlp_tools": "melusine/nlp_tools",
        "melusine.prepare_email": "melusine/prepare_email",
        "melusine.summarizer": "melusine/summarizer",
        "melusine.models": "melusine/models",
        "melusine.data": "melusine/data",
        "melusine.connectors": "melusine/connectors",
    },
    packages=[
        "melusine",
        "melusine.config",
        "melusine.utils",
        "melusine.nlp_tools",
        "melusine.prepare_email",
        "melusine.summarizer",
        "melusine.models",
        "melusine.data",
        "melusine.connectors",
    ],
    data_files=[
        ("config", conf_json_files),
        (
            "data",
            [
                "melusine/data/emails.csv",
                "melusine/data/emails_preprocessed.pkl",
                "melusine/data/emails_full.pkl",
            ],
        ),
    ],
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    extras_require=extras_require,
    url="https://github.com/MAIF/melusine",
    version='2.3.4',
    zip_safe=False,
)
