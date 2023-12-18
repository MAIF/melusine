[![pypi badge](https://img.shields.io/pypi/v/melusine.svg)](https://pypi.python.org/pypi/melusine)
[![Build & Test](https://github.com/MAIF/melusine/actions/workflows/main.yml/badge.svg?branch=master)](https://github.com/MAIF/melusine/actions/workflows/main.yml)[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Generic badge](https://img.shields.io/badge/python-3.8+-blue.svg)](https://shields.io/)

🎉 BREAKING : New major version **Melusine 3.0.0** is available 🎉  
Checkout the [documentation](https://maif.github.io/melusine/) and [tutorials](https://maif.github.io/melusine/tutorials/00_GettingStarted/) to get started.

<!-- <img src=`docs/_static/melusine.png` width=`200`/> -->
![](docs/_static/melusine.png)

- Free software: Apache Software License 2.0
- Documentation: [click here](https://maif.github.io/melusine/).
- Installation: `pip install melusine`.

## Overview


Melusine is a high-level library for emails processing that can be used to do:

- Email routing: Make sure emails are sent to the most appropriate destination.
- Prioritization: Ensure urgent emails are treated first.
- Summarization: Save time reading summaries instead of long emails.
- Filtering: Remove undesired emails.

Melusine facilitates the integration of deep learning frameworks (HuggingFace, Pytorch, Tensorflow, etc) 
deterministic rules (regex, keywords, heuristics) into a full email qualification workflow.

## Why melusine ?

The added value of melusine mainly resides in the following aspects:

- **Off-the-shelf features** :  melusine comes with a number of features that can be used straightaway
    - Segmenting messages in an email conversation
    - Tagging message parts (Email body, signatures, footers, etc)
    - Transferred email handling
- **Execution framework** : users can focus on the email qualification code and save time on the boilerplate code  
    - debug mode  
    - pipeline execution  
    - code parallelization
    - etc
- **Integrations** : the modular nature of melusine makes it easy to integrate with a variety of AI frameworks
  (HuggingFace, Pytorch, Tensorflow, etc)
- **Production ready** : melusine builds-up on the feedback from several years of running automatic email processing 
in production at MAIF.

## Getting started

Try one of our (tested!) [tutorials](https://maif.github.io/melusine/tutorials/00_GettingStarted/) to get started.

## Minimal example

- Load a fake email dataset
- Instantiate a built-in `MelusinePipeline`
- Run the qualification pipeline on the emails dataset

``` Python
    from melusine.data import load_email_data
    from melusine.pipeline import MelusinePipeline

    # Load an email dataset
    df = load_email_data()

    # Load a pipeline
    pipeline = MelusinePipeline.from_config("demo_pipeline")

    # Run the pipeline
    df = pipeline.transform(df)
```

The output is a qualified email dataset with columns such as:
- `messages`: List of individual messages present in each email.
- `emergency_result`: Flag to identify urgent emails.