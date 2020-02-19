import pandas as pd
import copy
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from melusine.utils.transformer_scheduler import TransformerScheduler

from melusine.prepare_email.manage_transfer_reply import check_mail_begin_by_transfer
from melusine.prepare_email.manage_transfer_reply import update_info_for_transfer_mail
from melusine.prepare_email.manage_transfer_reply import add_boolean_transfer
from melusine.prepare_email.manage_transfer_reply import add_boolean_answer

from melusine.prepare_email.build_historic import build_historic
from melusine.prepare_email.mail_segmenting import structure_email
from melusine.prepare_email.body_header_extraction import extract_last_body
from melusine.prepare_email.cleaning import clean_body
from melusine.prepare_email.cleaning import clean_header

from melusine.nlp_tools.phraser import Phraser
from melusine.nlp_tools.phraser import phraser_on_body
from melusine.nlp_tools.phraser import phraser_on_header
from melusine.nlp_tools.tokenizer import Tokenizer
from melusine.nlp_tools.embedding import Embedding
from melusine.summarizer.keywords_generator import KeywordsGenerator

from melusine.prepare_email.metadata_engineering import MetaExtension
from melusine.prepare_email.metadata_engineering import MetaDate
from melusine.prepare_email.metadata_engineering import Dummifier

from melusine.models.neural_architectures import cnn_model
from melusine.models.train import NeuralModel

from melusine.data.data_loader import load_email_data

body = "Bonjour\nThis is Melusine\nCordialement\nDev Team"
header = "Test integration Melusine"
body = "Bonjour\nThis is Melusine\nCordialement\nDev Team"
body = "Bonjour\nThis is Melusine\nCordialement\nDev Team"
body = "Bonjour\nThis is Melusine\nCordialement\nDev Team"



def test_classification():

    input_df = load_email_data()

    # ============== Train Phraser ==============
    phraser = Phraser(input_column='body',
                      threshold=10,
                      min_count=10)

    phraser.train(input_df)

    # ============== Define Transformer Schedulers ==============
    ManageTransferReply = TransformerScheduler(
        functions_scheduler=[
            (check_mail_begin_by_transfer, None, ['is_begin_by_transfer']),
            (update_info_for_transfer_mail, None, None),
            (add_boolean_answer, None, ['is_answer']),
            (add_boolean_transfer, None, ['is_transfer'])
        ]
    )

    Segmenting = TransformerScheduler(
        functions_scheduler=[
            (build_historic, None, ['structured_historic']),
            (structure_email, None, ['structured_body'])
        ]
    )
    LastBodyHeaderCleaning = TransformerScheduler(
        functions_scheduler=[
            (extract_last_body, None, ['last_body']),
            (clean_body, None, ['clean_body']),
            (clean_header, None, ['clean_header'])
        ]
    )

    # ============== Phraser ==============
    PhraserTransformer = TransformerScheduler(
        functions_scheduler=[
            (phraser_on_body, (phraser,), ['clean_body']),
            (phraser_on_header, (phraser,), ['clean_header'])
        ]
    )

    # ============== Tokenizer ==============
    tokenizer = Tokenizer(input_column="clean_body")

    # ============== Full Pipeline ==============
    PreprocessingPipeline = Pipeline([
        ('ManageTransferReply', ManageTransferReply),
        ('Segmenting', Segmenting),
        ('LastBodyHeaderCleaning', LastBodyHeaderCleaning),
        ('PhraserTransformer', PhraserTransformer),
        ('tokenizer', tokenizer)
    ])

    # ============== Transform input DataFrame ==============
    input_df = PreprocessingPipeline.transform(input_df)

    # ============== MetaData Pipeline ==============
    MetadataPipeline = Pipeline([
        ('MetaExtension', MetaExtension()),
        ('MetaDate', MetaDate()),
        ('Dummifier', Dummifier())
    ])

    # ============== Transform MetaData ==============
    df_meta = MetadataPipeline.fit_transform(input_df)

    # ============== Keywords Generator ==============
    keywords_generator = KeywordsGenerator(n_max_keywords=4)
    input_df = keywords_generator.fit_transform(input_df)

    # ============== Embeddings ==============
    pretrained_embedding = Embedding(input_column='clean_body',
                                     workers=1,
                                     min_count=5)
    pretrained_embedding.train(input_df)

    # ============== CNN Classifier ==============
    X = pd.concat([input_df['clean_body'], df_meta], axis=1)
    y = input_df['label']
    le = LabelEncoder()
    y = le.fit_transform(y)

    nn_model = NeuralModel(architecture_function=cnn_model,
                           pretrained_embedding=pretrained_embedding,
                           text_input_column="clean_body",
                           meta_input_list=['extension', 'dayofweek', 'hour', 'min'],
                           n_epochs=2)

    nn_model.fit(X, y)

    y_res = nn_model.predict(X)
    y_res = le.inverse_transform(y_res)

    # ============== Test dict compatibility ==============
    dict_emails = input_df.to_dict(orient='records')[0]
    dict_meta = MetadataPipeline.transform(dict_emails)
    dict_keywords = keywords_generator.transform(dict_emails)

    dict_input = copy.deepcopy(dict_meta)
    dict_input['clean_body'] = dict_emails['clean_body']

    dict_result = nn_model.predict(dict_input)

    assert True

