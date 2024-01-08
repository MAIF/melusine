# Welcome to melusine

<figure markdown>
  ![Melusine logo](_static/melusine.png){ align=center }
</figure>

## Overview

Discover Melusine, a comprehensive email processing library 
designed to optimize your email workflow. 
Leverage Melusine's advanced features to achieve:

- **Effortless Email Routing**: Ensure emails reach their intended destinations with high accuracy.
- **Smart Prioritization**: Prioritize urgent emails for timely handling and efficient task management.
- **Snippet Summaries**: Extract relevant information from lengthy emails, saving you precious time and effort.
- **Precision Filtering**: Eliminate unwanted emails from your inbox, maintaining focus and reducing clutter.

Melusine facilitates the integration of deep learning frameworks (HuggingFace, Pytorch, Tensorflow, etc), 
deterministic rules (regex, keywords, heuristics) into a full email qualification workflow.

## Why Choose Melusine ?

Melusine stands out with its combination of features and advantages:  

- **Out-of-the-box features** : Melusine comes with features such as
    - Segmenting an email conversation into individual messages
    - Tagging message parts (Email body, signatures, footers, etc)
    - Transferred email handling
- **Streamlined Execution** : Focus on the core email qualification logic 
while Melusine handles the boilerplate code, providing debug mode, pipeline execution, code parallelization, and more.
- **Flexible Integrations** : Melusine's modular architecture enables seamless integration with various AI frameworks, 
ensuring compatibility with your preferred tools.
- **Production ready** : Proven in the MAIF production environment, 
Melusine provides the robustness and stability you need.

## Email Segmentation Exemple

In the following example, an email is divided into two distinct messages 
separated by a transition pattern. 
Each message is then tagged line by line. 
This email segmentation can later be leveraged to enhance the performance of machine learning models.

???+ note "Message 1"

    <p style="text-align:left;"> Dear Kim
    <span style="float:right;background-color:#58D68D;"> HELLO</span>
    </p>
    <p style="text-align:left;"> Please find the details in the forwarded email.
    <span style="float:right;background-color:#F4D03F;"> BODY</span>
    </p>
    <p style="text-align:left;"> Best Regards
    <span style="float:right;background-color:#9C640C;"> GREETINGS</span>
    </p>
    <p style="text-align:left;"> Jo Kahn
    <span style="float:right;background-color:#EB984E;"> SIGNATURE</span>
    </p>

???+ note "Transition pattern"

    <p>Forwarded by jo@maif.fr on Monday december 12th</span>
    <span style="float:right;background-color:#D5DBDB;"> TRANSITION</span>
    </p>
    <p>From: alex@gmail.com
    <span style="float:right;background-color:#D5DBDB;"> TRANSITION</span>
    </p>
    <p>To: jo@maif.fr
    <span style="float:right;background-color:#D5DBDB;"> TRANSITION</span>
    </p>
    <p>Subject: New address
    <span style="float:right;background-color:#D5DBDB;"> TRANSITION</span>
    </p>

???+ note "Message 2"

    <p style="text-align:left;"> Dear Jo
    <span style="float:right;background-color:#58D68D;"> HELLO</span>
    </p>
    <p style="text-align:left;"> A new version of Melusine is about to be released.
    <span style="float:right;background-color:#F4D03F;"> BODY</span>
    </p>
    <p style="text-align:left;"> Feel free to test it and send us feedbacks!
    <span style="float:right;background-color:#F4D03F;"> BODY</span>
    </p>
    <p style="text-align:left;"> Thank you for your help.
    <span style="float:right;background-color:#A93226;"> THANKS</span>
    </p>
    <p style="text-align:left;"> Cheers
    <span style="float:right;background-color:#9C640C;"> GREETINGS</span>
    </p>
    <p style="text-align:left;"> Alex Leblanc
    <span style="float:right;background-color:#EB984E;"> SIGNATURE</span>
    </p>
    <p style="text-align:left;"> 55 Rue du Faubourg Saint-Honoré
    <span style="float:right;background-color:#EB984E;"> SIGNATURE</span>
    </p>
    <p style="text-align:left;"> 75008 Paris
    <span style="float:right;background-color:#EB984E;"> SIGNATURE</span>
    </p>
    <p style="text-align:left;"> Sent from my iPhone
    <span style="float:right;background-color:#6E2C00;"> FOOTER</span>
    </p>


## Getting Started

Get started with melusine following our (tested!) tutorials:

* [Getting Started](tutorials/00_GettingStarted.md){target=_blank}

* [MelusinePipeline](tutorials/01_MelusinePipeline.md){target=_blank}

* [MelusineTransformers](tutorials/02_MelusineTransformers.md){target=_blank}

* [MelusineRegex](tutorials/03_MelusineRegex.md){target=_blank}

* [ML models](tutorials/04_UsingModels.md){target=_blank}

* [MelusineDetector](tutorials/05a_MelusineDetectors.md){target=_blank}

* [Configurations](tutorials/06_Configurations.md){target=_blank}

* [Basic Classification](tutorials/07_BasicClassification.md){target=_blank}


With Melusine, you're well-equipped to transform your email handling, streamlining processes, maximizing efficiency, 
and enhancing overall productivity.