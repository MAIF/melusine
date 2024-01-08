# Advanced Melusine Detectors

This tutorial presents the advanced features of the `MelusineDetector` class:

- Debug mode
- Row wise methods vs DataFrame wise methods
- Custom transform methods

## Debug mode

`MelusineDetector` are designed to be easily debugged. For that purpose, the `pre-detect`/`detect`/`post-detect` methods all have a `debug_mode` argument. The debug mode is activated by setting the `debug` attribute of a dataframe to `True`.

```Python hl_lines="3"
import pandas as pd
df = pd.DataFrame({"bla": [1, 2, 3]})
df.debug = True
```

!!! warning
    Debug mode activation is backend dependent. With a DictBackend, tou should use `my_dict["debug"] = True`

When debug mode is activated, a column named `DETECTOR_NAME_debug` containing an empty dictionary is automatically created.
Populating this debug dict with debug info is then left to the user's responsibility. 

Example of a detector with debug data:

```Python hl_lines="21 22 37-53"
--8<--
docs/docs_src/MelusineDetectors/tutorial003.py:detector
--8<--
```

In the end, an extra column is created containing debug data:

|    | virus_result   | debug_virus                                                                                                                                                       |
|---:|:---------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 | True           | {'detection_input': '...', 'positive_match_data': {'result': True, 'match_text': 'virus'}, 'negative_match_data': {'result': False, 'match_text': None}}          |
|  1 | False          | {'detection_input': '...', 'positive_match_data': {'result': False, 'match_text': None}, 'negative_match_data': {'result': False, 'match_text': None}}            |
|  2 | True           | {'detection_input': '...', 'positive_match_data': {'result': True, 'match_text': 'virus'}, 'negative_match_data': {'result': False, 'match_text': None}}          |
|  3 | False          | {'detection_input': '...', 'positive_match_data': {'result': True, 'match_text': 'virus'}, 'negative_match_data': {'result': True, 'match_text': 'corona virus'}} |e          | {'detection_input': 'test\ncorona virus is annoying', 'positive_match_data': {'result': True, 'match_text': 'virus'}, 'negative_match_data': {'result': True, 'match_text': 'corona virus'}} |

## Row Methods VS Dataframe Methods

There are two ways to use the `pre-detect`/`detect`/`post-detect` methods:

- **Row wise**: The method works on a single row of a `DataFrame`. In that case, a map-like method is used to apply it on an entire dataframe (typically [`pandas.DataFrame.apply`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html) is used with the PandasBackend).
- **Dataframe wise**: The method works directly on the entire DataFrame.

!!! tip
    Using row wise methods make your code backend independent. You may 
    switch from a `PandasBackend` to a `DictBackend` at any time. 
    The `PandasBackend` also supports multiprocessing for row wise methods.

To use row wise methods, you just need to name the first parameter of "row". Otherwise, dataframe wise transformations are used.

Example of a Detector with dataframe wise method (works with a PandasBackend only).

```Python hl_lines="22 28 39"
--8<--
docs/docs_src/MelusineDetectors/tutorial002.py:detector
--8<--
```

## Custom Transform Methods

If you are not happy with the `pre_detect`/`detect`/`post_detect` transform methods, you may: 

- Use custom template methods.
- Use regular pipeline steps (not inheriting from the `MelusineDetector` class).

- In this example, the `prepare`/`run` custom transform methods are used instead of the default `pre_detect`/`detect`/`post_detect`.

```Python
--8<--
docs/docs_src/MelusineDetectors/tutorial004.py:detector
--8<--
```

To configure custom transform methods you need to: 

- Inherit from the `melusine.base.BaseMelusineDetector` class.
- Define the `transform_methods` property.

The `transform` method will now call `prepare` and `run`.

```Python
--8<--
docs/docs_src/MelusineDetectors/tutorial004.py:run
--8<--
```

We can check that the `run` method was indeed called.

|    | input_col   |   output_col |
|---:|:------------|-------------:|
|  0 | test1       |        12345 |
|  1 | test2       |        12345 |
