#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['pandas>=0.25.0',
                'scikit-learn>=0.19.0',
                'gensim>=3.3.0',
                'keras>=2.2.0',
                'tqdm>=4.34',
                'tensorflow>=1.10.0,<=1.13.1',
                'unidecode',
                'joblib'
                ]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Sacha Samama, Tom Stringer, Antoine Simoulin, Benoit Lebreton",
    author_email='ssamama@quantmetry.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Melusine is a high-level package for french emails preprocessing, classification and feature extraction, written in Python.",
    entry_points={},
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='melusine',
    name='melusine',
    package_dir={
        'melusine': 'melusine',
        'melusine.config': 'melusine/config',
        'melusine.utils': 'melusine/utils',
        'melusine.nlp_tools': 'melusine/nlp_tools',
        'melusine.prepare_email': 'melusine/prepare_email',
        'melusine.summarizer': 'melusine/summarizer',
        'melusine.models': 'melusine/models',
        'melusine.data': 'melusine/data'
    },
    packages=['melusine', 'melusine.config', 'melusine.utils',
              'melusine.nlp_tools', 'melusine.prepare_email',
              'melusine.summarizer', 'melusine.models',
              'melusine.data'],
    data_files=[('config', ['melusine/config/conf.json']),
                ('data', ['melusine/data/emails.csv'])],
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/MAIF/melusine',
    version='1.9.4',
    zip_safe=False,
)
