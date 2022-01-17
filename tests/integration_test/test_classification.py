import pandas as pd
import copy
import os
import ast
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from melusine.models.neural_architectures import cnn_model
from melusine.models.train import NeuralModel
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
from melusine.nlp_tools.tokenizer import Tokenizer
from melusine.nlp_tools.embedding import Embedding
from melusine.summarizer.keywords_generator import KeywordsGenerator

from melusine.prepare_email.metadata_engineering import MetaExtension
from melusine.prepare_email.metadata_engineering import MetaDate
from melusine.prepare_email.metadata_engineering import MetaAttachmentType
from melusine.prepare_email.metadata_engineering import Dummifier

from melusine.data.data_loader import load_email_data

# Prevent GPU usage
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def test_classification():

    input_df = load_email_data()

    # ============== Define Transformer Schedulers ==============
    manageTransferReply = TransformerScheduler(
        functions_scheduler=[
            (check_mail_begin_by_transfer, None, ["is_begin_by_transfer"]),
            (update_info_for_transfer_mail, None, None),
            (add_boolean_answer, None, ["is_answer"]),
            (add_boolean_transfer, None, ["is_transfer"]),
        ]
    )

    segmenting = TransformerScheduler(
        functions_scheduler=[
            (build_historic, None, ["structured_historic"]),
            (structure_email, None, ["structured_body"]),
        ]
    )
    lastBodyHeaderCleaning = TransformerScheduler(
        functions_scheduler=[
            (extract_last_body, None, ["last_body"]),
            (clean_body, None, ["clean_body"]),
            (clean_header, None, ["clean_header"]),
        ]
    )

    # ============== Tokenizer ==============
    tokenizer = Tokenizer(input_column="clean_body")

    # ============== Phraser ==============
    phraser = Phraser(threshold=10, min_count=10)

    # ============== Full Pipeline ==============
    preprocessingPipeline = Pipeline(
        [
            ("ManageTransferReply", manageTransferReply),
            ("Segmenting", segmenting),
            ("LastBodyHeaderCleaning", lastBodyHeaderCleaning),
            ("tokenizer", tokenizer),
            ("phraser", phraser),
        ]
    )

    # ============== Transform input DataFrame ==============
    input_df = preprocessingPipeline.fit_transform(input_df)

    # ============== MetaData Pipeline ==============
    metadataPipeline = Pipeline(
        [
            ("MetaExtension", MetaExtension()),
            ("MetaDate", MetaDate()),
            ("MetaAttachmentType", MetaAttachmentType()),
            ("Dummifier", Dummifier()),
        ]
    )

    # ============== Transform MetaData ==============
    input_df["attachment"] = input_df["attachment"].apply(ast.literal_eval)
    df_meta = metadataPipeline.fit_transform(input_df)

    # ============== Keywords Generator ==============
    keywords_generator = KeywordsGenerator(n_max_keywords=4)
    input_df = keywords_generator.fit_transform(input_df)

    # ============== Embeddings ==============
    pretrained_embedding = Embedding(tokens_column="tokens", workers=1, min_count=5)
    pretrained_embedding.train(input_df)

    # ============== CNN Classifier ==============
    X = pd.concat([input_df["clean_body"], df_meta], axis=1)
    y = input_df["label"]
    le = LabelEncoder()
    y = le.fit_transform(y)

    nn_model = NeuralModel(
        architecture_function=cnn_model,
        pretrained_embedding=pretrained_embedding,
        text_input_column="clean_body",
        meta_input_list=[
            "extension",
            "dayofweek",
            "hour",
            "min",
            "attachment_type",
        ],
        n_epochs=2,
        seq_size=8,
        batch_size=8,
    )

    nn_model.fit(X, y)

    y_res = nn_model.predict(X)
    le.inverse_transform(y_res)

    # ============== Test dict compatibility ==============
    dict_emails = input_df.to_dict(orient="records")[0]
    dict_meta = metadataPipeline.transform(dict_emails)
    keywords_generator.transform(dict_emails)

    dict_input = copy.deepcopy(dict_meta)
    dict_input["clean_body"] = dict_emails["clean_body"]

    nn_model.predict(dict_input)

    assert True
