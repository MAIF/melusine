#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = [
    "tensorflow>=2.5.0",
    "pandas>=1.0",
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

# Test dependencies
setup_requirements = ["pytest-runner"]
test_requirements = transformers_requirements + ["pytest"]


# Ex: Install all dependencies with ``pip install melusine[all]`
extras_require = {
    "exchange": exchange_requirements,
    "transformers": transformers_requirements,
    "viz": viz_requirements,
}
all_requirements = list(set([y for x in extras_require.values() for y in x]))
extras_require["all"] = all_requirements


setup(
    author="Sacha Samama, Tom Stringer, Antoine Simoulin, Benoit Lebreton, Tiphaine Fabre, Hugo Perrier",
    author_email="tiphaine.fabre@maif.fr",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
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
        ("config", ["melusine/config/tokenizer_conf.json"]),
        ("config", ["melusine/config/conf.json"]),
        ("config", ["melusine/config/names.json"]),
        ("data", ["melusine/data/emails.csv"]),
        ("data", ["melusine/data/emails_preprocessed.csv"]),
        ("data", ["melusine/data/emails_full.csv"]),
    ],
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    extras_require=extras_require,
    url="https://github.com/MAIF/melusine",
    version="2.3.1",
    zip_safe=False,
)
