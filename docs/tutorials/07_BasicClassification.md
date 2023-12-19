# Zero Shot Classification

Machine Learning is commonly used to classify data into pre-defined categories. 

``` mermaid
---
title: Email classification
---
flowchart LR
    Input[[Email]] --> X(((Classifier)))
    X --> A(Car)
    X --> B(Boat)
    X --> C(Housing)
    X --> D(Health)
```

Typically, to reach high classification performance, 
models need to be trained on context specific labeled data. 
Zero-shot classification is a type of classification that 
uses a pre-trained model and does not require further training on context specific data.

## Tutorial intro
In this tutorial we want to detect insatisfaction in an email dataset. 
Let's create a basic dataset:
```Python
--8<--
docs/docs_src/BasicClassification/tutorial001.py:create_dataset
--8<--
```

|    | header                       | body                                                                               |
|---:|:-----------------------------|:-----------------------------------------------------------------------------------|
|  0 | Dossier 123456               | Merci beaucoup pour votre gentillesse et votre écoute !                            |
|  1 | Réclamation (Dossier 987654) | Bonjour, je ne suis pas satisfait de cette situation, répondez-moi rapidement svp! |


## Classify with Zero-Shot-Classification

The `transformers` library makes it really simple to use pre-trained models for zero shot classification.

```Python
--8<--
docs/docs_src/BasicClassification/tutorial001.py:transformers
--8<--
```

The classifier returns a score for the "positif" and "négatif" label for each input text:

```Json
[
    {
        'sequence': "Quelle belle journée aujourd'hui",
        'labels': ['positif', 'négatif'],
        'scores': [0.95, 0.05]
    },
    {
        'sequence': 'La marée est haute',
        'labels': ['positif', 'négatif'],
        'scores': [0.76, 0.24]
    },
    {'sequence': 'Ce film est une catastrophe, je suis en colère',
     'labels': ['négatif', 'positif'],
     'scores': [0.97, 0.03]
     }
]
```


## Implement a Dissatisfaction detector

A full email processing pipeline could contain multiple models. 
Melusine uses the MelusineDetector template class to standardise how models are integrated into a pipeline.

```Python
--8<--
docs/docs_src/BasicClassification/tutorial001.py:detector_init
--8<--
```

The `pre_detect` method assembles the text that we want to use for classification.

```Python
--8<--
docs/docs_src/BasicClassification/tutorial001.py:pre_detect
--8<--
```

The `detect` method runs the classification model on the text.

```Python
--8<--
docs/docs_src/BasicClassification/tutorial001.py:detect
--8<--
```

The `post_detect` method applies a threshold on the prediction score to determine the detection result.

```Python
--8<--
docs/docs_src/BasicClassification/tutorial001.py:post_detect
--8<--
```

On top of that, the detector takes care of building debug data to make the result explicable.

## Run detection

Putting it all together, we run the detector on the input dataset.

```Python
--8<--
docs/docs_src/BasicClassification/tutorial001.py:run
--8<--
```

As a result, we get a new column `dissatisfaction_result` with the detection result. 
We could have detection details by running the detector in debug mode.

|    | header                       | body                                                                               | dissatisfaction_result   |
|---:|:-----------------------------|:-----------------------------------------------------------------------------------|:-------------------------|
|  0 | Dossier 123456               | Merci beaucoup pour votre gentillesse et votre écoute !                            | False                    |
|  1 | Réclamation (Dossier 987654) | Bonjour, je ne suis pas satisfait de cette situation, répondez-moi rapidement svp! | True                     |