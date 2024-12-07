# Use pre-trained models from HuggingFace in the Melusine framework


> The Hugging Face library has become an invaluable resource in the data science field, offering easy-to-use models that excel across a variety of natural language processing (NLP) tasks.

# How to leverage these models within the Melusine framework to build:

* A powerful processor using model embeddings.
* An intelligent detector utilizing fine-tuned model layers.

Transformers-based models from Hugging Face can significantly enhance detection capabilities and act as a complementary approach to strengthen prediction results.

## How to Choose and Use Models

The selection of a model depends on the specific detection task. For example:

    **Sentiment detection in French text/emails:** 
    Suitable models include: camembert-base, distil-camembert-base, or distil-camembert-base.
    These models can be seamlessly integrated into your workflow to enhance predictions and optimize detection outcomes.



# Implementing solution : distil-camembert Models 
As usual , the detector can be implemented this way , inheriting the MelusineTransformerDetector detector base class 

```python
class DissatisfactionDetector(MelusineTransformerDetector):
    """
    Class to detect emails containing only dissatisfaction text.

    Ex:
    Merci à vous,
    Cordialement
    """

    # Class constants
    BODY_PART: str = "BODY"
    DISSATISFACTION_PART: str = "DISSATISFACTION"

    # Intermediate columns
    THANKS_TEXT_COL: str = "thanks_text"
    THANKS_PARTS_COL: str = "thanks_parts"
    HAS_BODY: str = "has_body"
    THANKS_MATCH_COL: str = "thanks_match"

    def __init__(
        self,
        messages_column: str = "messages",
        name: str = "dissatisfaction",
    ) -> None:
        """
        Attributes initialization.

        Parameters
        ----------
        messages_column: str
            Name of the column containing the messages.

        name: str
            Name of the detector.
        """

        # Input columns
        self.messages_column = messages_column
        input_columns: List[str] = [self.messages_column]

        # Output columns
        self.result_column = f"{name}_result"
        output_columns: List[str] = [self.result_column]

        # Detection regex
        self.thanks_regex: MelusineRegex = ThanksRegex()

        super().__init__(
            name=name,
            input_columns=input_columns,
            output_columns=output_columns,
        )
        self.complex_regex_key: str
```

> The pre_detect method allows to preprocess the data into the type of model inputs . 

```python
def pre_detect(self, row: MelusineItem, debug_mode: bool = False) -> MelusineItem:
    """
    Extract text to analyse.

    Parameters
    ----------
    row: MelusineItem
        Content of an email.
    debug_mode: bool
        Debug mode activation flag.

    Returns
    -------
    row: MelusineItem
        Updated row.
    """
    # Check if a BODY part is present in the last message
    has_body: bool = row[self.messages_column][0].has_tags(
        target_tags={self.BODY_PART}, stop_at={self.GREETINGS_PART}
    )

    # Extract the DISSATISFACTION part in the last message
    dissatisfaction_parts: List[Tuple[str, str]] = row[self.messages_column][
        0
    ].extract_parts(target_tags={self.DISSATISFACTION_PART})

    # Compute DISSATISFACTION text
    if not dissatisfaction_parts:
        dissatisfaction_text: str = ""
    else:
        dissatisfaction_text = "\n".join(x[1] for x in dissatisfaction_parts)

    # Save debug data
    if debug_mode:
        debug_dict = {
            self.DISSATISFACTION_PARTS_COL: dissatisfaction_parts,
            self.DISSATISFACTION_TEXT_COL: dissatisfaction_text,
            self.HAS_BODY: has_body,
        }
        row[self.debug_dict_col].update(debug_dict)

    # Create new columns
    row[self.DISSATISFACTION_TEXT_COL] = dissatisfaction_text
    row[self.HAS_BODY] = has_body

    return row
```
> The detection method can be one of the following three : 
    * deterministic only : using Melusine_regex :
    * Machine learning based only : using HF models
    * both are combined to one final output : the detection result 


* A dissatisfaction_regex MUST BE CREATED WITH DIFFERENT REGEX USEFUL TO DETECT DISSATISFACTION 

```python 
from typing import Dict, List, Optional, Union
from melusine.base import MelusineRegex


class DissatisfactionRegex(MelusineRegex):
    """
    Detect thanks patterns such as "merci".
    """

    @property
    def positive(self) -> Union[str, Dict[str, str]]:
        """
        Define regex patterns required to activate the MelusineRegex.

        Returns:
            _: Regex pattern or dict of regex patterns.
        """

        return r"\b(j'en ai marre|c'est nul|trop déçu|décevant|inadmissible|insupportable|intolérable|honteux|lamentable|catastrophe)\b"

    @property
    def neutral(self) -> Optional[Union[str, Dict[str, str]]]:
        """
        Define regex patterns to be ignored when running detection.

        Returns:
            _: Regex pattern or dict of regex patterns.
        """
        return None

    @property
    def negative(self) -> Optional[Union[str, Dict[str, str]]]:
        """
        Define regex patterns prohibited to activate the MelusineRegex.

        Returns:
            _: Regex pattern or dict of regex patterns.
        """
        return None

    @property
    def match_list(self) -> List[str]:
        """
        List of texts that should activate the MelusineRegex.

        Returns:
            _: List of texts.
        """
        return [
            "complétement insatisfait de ce que vous faites",
        ]

    @property
    def no_match_list(self) -> List[str]:
        """
        List of texts that should NOT activate the MelusineRegex.

        Returns:
            _: List of texts.
        """
        return []
```
After constructing the DissatisfactionRegex class , the by_regex_detect method could be defined 
```python
def by_regex_detect(self, row: MelusineItem, debug_mode: bool = False) -> MelusineItem:
    """
    Use regex to detect dissatisfaction.

    Parameters
    ----------
    row: MelusineItem
        Content of an email.
    debug_mode: bool
        Debug mode activation flag.

    Returns
    -------
    row: MelusineItem
        Updated row.
    """
    debug_info: Dict[str, Any] = {}

    text: str = row[self.DISSATISFACTION_TEXT_COL]

    detection_data = self.dissatisfaction_regex(text)
    detection_result = detection_data[self.dissatisfaction_regex.MATCH_RESULT]

    # Save debug data
    if debug_mode:
        debug_info[self.dissatisfaction_regex.regex_name] = detection_data
        row[self.debug_dict_col].update(debug_info)

    # Create new columns
    row[self.DISSATISFACTION_BY_REGEX_MATCH_COL] = detection_result

    return row
```

## The Machine Learning Approach to Detect Dissatisfaction: Two Methods
    * Using a Pre-trained Model Directly
        The distil-camembert-base model can be loaded directly from the Hugging Face platform, along with its tokenizer, for immediate use in detecting dissatisfaction.

    * Fine-tuning the Model
        A pre-trained model can be fine-tuned using various methods, including:

        The Hugging Face Trainer API or PyTorch Lightning.
    
> Fine-tuning approaches:
    1- Full Fine-tuning: Updates all layers of the model in an autoregressive manner.
    2-  LoRA's PEFT (Parameter-Efficient Fine-Tuning): A more efficient and optimized method that reduces computational cost while achieving excellent results.

Fine-tuning allows customization of the model for specific tasks, improving its performance on datasets relevant to dissatisfaction detection.

> Why distil-camembert-base?
Numerous studies and practical implementations have demonstrated that distil-camembert-base is a highly effective model for sentiment analysis and detecting dissatisfaction, particularly in tasks involving French text.
The model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax


def load_hfmodel(self, model_name="distilcamembert-base") -> None:
    """
    GET Distil-camembert-base from HF
    Parameters
    ----------
    row: MelusineItem
        Content of an email.
    debug_mode: bool
        Debug mode activation flag.

    Returns
    -------
    row: MelusineItem
        Updated row.
    """

    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=5
    )  # Adjust num_labels for your classification task

    def predict_fn(self, text) -> List:
        """
        Apply model and get prediction
        Parameters
        ----------
        row: MelusineItem
            Content of an email.
        debug_mode: bool
            Debug mode activation flag.

        Returns
        -------
        row: MelusineItem
            Updated row.
        """

        inputs = self.tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        )
        # Forward pass through the model
        outputs = self.model(**inputs)

        # Convert logits to probabilities using softmax
        probs = softmax(logits, dim=1)

        # Get predictions (the class index with the highest probability)
        predictions = probs.argmax(dim=1).tolist()

        # Get confidence scores for the predicted classes
        scores = probs.max(dim=1).values.tolist()
        return predictions, scores

    def by_ml_detect(self, row: MelusineItem, debug_mode: bool = False) -> MelusineItem:
        """
        Use machine learning model to detect dissatisfaction.

        Parameters
        ----------
        row: MelusineItem
            Content of an email.
        debug_mode: bool
            Debug mode activation flag.

        Returns
        -------
        row: MelusineItem
            Updated row.
        """
        debug_info: Dict[str, Any] = {}
        (
            row[self.DISSATISFACTION_ML_MATCH_COL],
            row[self.DISSATISFACTION_ML_SCORE_COL],
        ) = self.predict_fn(row[self.DISSATISFACTION_TEXT_COL])
        # Save debug data
        if debug_mode:
            debug_info[self.DISSATISFACTION_ML_MATCH_COL] = row[
                self.DISSATISFACTION_ML_MATCH_COL
            ]
            debug_info[self.DISSATISFACTION_ML_SCORE_COL] = row[
                self.DISSATISFACTION_ML_SCORE_COL
            ]
            row[self.debug_dict_col].update(debug_info)
        return row
```


> The final detection result could be defined in the **post_detect** method using a predefined condition. 
> [! Example ]
> condition :  by_regex_detect OR (by_ml_detect and by_ml_detect.score > .9)
 

```python
def post_detect(self, row: MelusineItem, debug_mode: bool = False) -> MelusineItem:
    """
    Apply final eligibility rules.

    Parameters
    ----------
    row: MelusineItem
        Content of an email.
    debug_mode: bool
        Debug mode activation flag.

    Returns
    -------
    row: MelusineItem
        Updated row.
    """

    # Match on thanks regex & Does not contain a body
    row[self.result_column] = (
        row[self.DISSATISFACTION_ML_SCORE_COL] > 0.9
        and row[self.DISSATISFACTION_ML_MATCH_COL]
    ) or row[self.DISSATISFACTION_BY_REGEX_MATCH_COL]

    return row
```