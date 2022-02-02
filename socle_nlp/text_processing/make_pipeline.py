from socle_nlp.utils.load_config import get_default_config
from socle_nlp.text_processing.all_transformers import *
from socle_nlp.utils.socle_nlp_pipeline import SocleNlpPipeline

AVAILABLE_TRANSFORMERS = {'ToStr': ToStrTransformer,
                          'RemoveLineBreak': RemoveLineBreakTransformer,
                          'RemoveMultipleSpace': RemoveMultiSpaceTransformer,
                          'RemoveAccent': RemoveAccentTransformer,
                          'Strip': StripTransformer,
                          'ToLower': ToLowerTransformer,
                          'RemovePunctuation': RemovePunctTransformer}


def check_config_pipeline(config_pipeline: dict) -> None:
    if not isinstance(config_pipeline, dict):
        raise Exception(f"Bad format")


def check_in_out(in_out: list):
    if len(in_out) != 2:
        raise Exception(f"Bad format")
    else:
        return in_out[0], in_out[1]


def build_sequence(sequence: dict) -> List:
    transformers_seq = []
    for trf_name, in_out in sequence.items():
        if trf_name in AVAILABLE_TRANSFORMERS:
            trf_class = AVAILABLE_TRANSFORMERS[trf_name]
            in_col, out_col = check_in_out(in_out)
            transformers_seq.append((trf_name, trf_class(in_col, out_col)))
        else:
            raise Exception(f"{trf_name} not implemented")
    return transformers_seq


def make_pipeline(pipeline: str, file: str = "config_pipeline.json") -> SocleNlpPipeline:
    config = get_default_config(file)
    config_pipeline = config[pipeline]
    check_config_pipeline(config_pipeline)
    return SocleNlpPipeline(sequence=build_sequence(config_pipeline))


def main() -> None:
    import pandas as pd
    from sklearn.pipeline import Pipeline
    data = pd.DataFrame(columns=['text'])
    data['text'] = ['Ca fonctionne\n\n Ca  va ??', 'Ca fonctionne    Bien HéHé']
    data['tex2'] = ['Ca fonctionne\n\n Ca va ??', 'Ca fonctionne Bien']
    
    test = Pipeline([('first pipe', make_pipeline('test1')),
                     ('second pipe', make_pipeline('test2'))])
    print(test.fit_transform(data))
    print(data)


if __name__ == "__main__":
    main()
