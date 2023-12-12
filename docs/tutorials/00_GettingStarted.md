# Getting started with Melusine

Let's run **emergency detection** with melusine :

* Load a fake email dataset
* Load a demonstration pipeline
* Run the pipeline  
    * Apply email cleaning transformations  
    * Apply emergency detection

## Input data

Email datasets typically contain information about:

- Email sender
- Email recipients
- Email subject/header
- Email body
- Attachments data

The present tutorial only makes use of the **body** and **header** data.

|    | body                             | header      |
|:---|:---------------------------------|:------------|
| 0  | This is an ëmèrgénçy             | Help        |
| 1  | How is life ?                    | Hey !       |
| 2  | Urgent update about Mr. Annoying | Latest news |
| 3  | Please call me now               | URGENT      |

## Code

A typical code for a melusine-based application looks like this :

```Python
--8<--
docs_src/GettingStarted/tutorial001.py:simple_pipeline
--8<--
```

1. This tutorial uses one of the default pipeline configuration `demo_pipeline`. Melusine users will typically define their own pipeline configuration.
   See more in the [Configurations tutorial](06_Configurations.md){target=_blank}

## Output data

The pipeline created extra columns in the dataset.
Some columns are temporary variables required by detectors (ex: `normalized_body`)
and some are detection results with direct business value (ex: `emergency_result`).

|    | body                             | header      | normalized_body             | emergency_result   |
|:---|:---------------------------------|:------------|:---------------------------------|:-------------------|
| 0  | This is an ëmèrgénçy             | Help        | This is an emergency             | True               |
| 1  | How is life ?                    | Hey !       | How is life ?                    | False              |
| 2  | Urgent update about Mr. Annoying | Latest news | Urgent update about Mr. Annoying | False              |
| 3  | Please call me now               | URGENT      | Please call me now               | True               |

## Pipeline steps

Illustration of the pipeline used in the present tutorial :

``` mermaid
---
title: Demonstration pipeline
---
flowchart LR
    Input[[Email]] --> A(Cleaner)
    A(Cleaner) --> C(Normalizer)
    C --> F(Emergency\nDetector)
    F --> Output[[Qualified Email]]
```

* `Cleaner` : Cleaning transformations such as uniformization of line breaks (`\r\n` -> `\n`)
* `Normalizer` : Text normalisation to delete/replace non utf8 characters (`éöà` -> `eoa`)
* `EmergencyDetector` : Detection of urgent emails


!!! info
    This demonstration pipeline is kept minimal but typical pipelines include more complex preprocessing and a variety of detectors.
    For example, pipelines may contain:

    - Email Segmentation : Split email conversation into unitary messages
    - ContentTagging : Associate tags (SIGNATURE, FOOTER, BODY) to parts of messages
    - Appointment detection : For exemple, detect "construction work will take place on 01/01/2024" as an appointment email.
    - More on preprocessing in the [MelusineTransformers tutorial](02_MelusineTransformers.md){target=_blank}
    - More on detectors in the [MelusineDetector tutorial](05a_MelusineDetectors.md){target=_blank}


## Debug mode

End users typically want to know what lead melusine to a specific detection result. The debug mode generates additional explainability info.

```Python
--8<--
docs_src/GettingStarted/tutorial002.py:debug_pipeline
--8<--
```


A new column `debug_emergency` is created.

|    | ... | emergency_result   | debug_emergency   |
|:---|:----|:-------------------|:------------------|
| 0  | ... | True               | [details_below]   |
| 1  | ... | False              | [details_below]   |
| 2  | ... | False              | [details_below]   |
| 3  | ... | True               | [details_below]   |

Inspecting the debug data gives a lot of info:

- `text` : Effective text considered for detection.
- `EmergencyRegex` : melusine used an `EmergencyRegex` object to run detection.
- `match_result` : The `EmergencyRegex` did not match the text
- `positive_match_data` : The `EmergencyRegex` matched **positively** the text pattern "Urgent" (Required condition)
- `negative_match_data` : The `EmergencyRegex` matched **negatively** the text pattern "Mr. Annoying" (Forbidden condition)
- `BLACKLIST` : Detection groups can be defined to easily link a matching pattern to the corresponding regex. DEFAULT is used if no detection group is specified.


```Python
# print(df.iloc[2]["debug_emergency"])
{
  'text': 'Latest news\nUrgent update about Mr. Annoying'},
  'EmergencyRegex': {
    'match_result': False,
    'negative_match_data': {
      'BLACKLIST': [
        {'match_text': 'Mr. Annoying', 'start': 32, 'stop': 44}
      ]},
    'neutral_match_data': {},
    'positive_match_data': {
      'DEFAULT': [
        {'match_text': 'Urgent', 'start': 12, 'stop': 18}
      ]
    }
  }
```
