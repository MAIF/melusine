from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Bidirectional

from melusine.models.attention_model import (
    PositionalEncoding,
    TransformerEncoderLayer,
)


def cnn_model(
    embedding_matrix_init,
    ntargets,
    seq_max,
    nb_meta,
    loss="categorical_crossentropy",
    activation="softmax",
):
    """Pre-defined architecture of a CNN model.

    Parameters
    ----------
    embedding_matrix_init : np.array,
        Pretrained embedding matrix.

    ntargets : int, optional
        Dimension of model output.
        Default value, 18.

    seq_max : int, optional
        Maximum input length.
        Default value, 100.

    nb_meta : int, optional
        Dimension of meta data input.
        Default value, 252.

    loss : str, optional
        Loss function for training.
        Default value, 'categorical_crossentropy'.

    activation : str, optional
        Activation function.
        Default value, 'softmax'.

    Returns
    -------
    Model instance
    """

    text_input = Input(shape=(seq_max,), dtype="int32")

    x = Embedding(
        input_dim=embedding_matrix_init.shape[0],
        output_dim=embedding_matrix_init.shape[1],
        input_length=seq_max,
        weights=[embedding_matrix_init],
        trainable=True,
    )(text_input)
    x = Conv1D(200, 2, padding="same", activation="linear", strides=1)(x)
    x = SpatialDropout1D(0.15)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Conv1D(250, 2, padding="same", activation="linear", strides=1)(x)
    x = SpatialDropout1D(0.15)(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Dropout(0.15)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(250, activation="linear")(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Dense(150, activation="linear")(x)
    x = Dropout(0.15)(x)
    x = LeakyReLU(alpha=0.05)(x)

    if nb_meta == 0:
        inputs = text_input
        concatenate_2 = x
    else:
        Meta_input = Input(shape=(nb_meta,), dtype="float32")
        inputs = [text_input, Meta_input]

        concatenate_1 = Meta_input
        y = Dense(150, activation="linear")(concatenate_1)
        y = Dropout(0.2)(y)
        y = LeakyReLU(alpha=0.05)(y)
        y = Dense(100, activation="linear")(y)
        y = Dropout(0.2)(y)
        y = LeakyReLU(alpha=0.05)(y)
        y = Dense(80, activation="linear")(y)
        y = Dropout(0.2)(y)
        y = LeakyReLU(alpha=0.05)(y)
        concatenate_2 = Concatenate(axis=1)([x, y])

    z = Dense(200, activation="linear")(concatenate_2)
    z = Dropout(0.2)(z)
    z = LeakyReLU(alpha=0.05)(z)
    z = Dense(100, activation="linear")(z)
    z = Dropout(0.2)(z)
    z = LeakyReLU(alpha=0.05)(z)
    outputs = Dense(ntargets, activation=activation)(z)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=Adam(), loss=loss, metrics=["accuracy"])

    return model


def rnn_model(
    embedding_matrix_init,
    ntargets=18,
    seq_max=100,
    nb_meta=252,
    loss="categorical_crossentropy",
    activation="softmax",
):
    """Pre-defined architecture of a RNN model.

    Parameters
    ----------
    embedding_matrix_init : np.array,
        Pretrained embedding matrix.

    ntargets : int, optional
        Dimension of model output.
        Default value, 18.

    seq_max : int, optional
        Maximum input length.
        Default value, 100.

    nb_meta : int, optional
        Dimension of meta data input.
        Default value, 252.

    loss : str, optional
        Loss function for training.
        Default value, 'categorical_crossentropy'.

    activation : str, optional
        Activation function.
        Default value, 'softmax'.

    Returns
    -------
    Model instance
    """
    text_input = Input(shape=(seq_max,), dtype="int32")

    x = Embedding(
        input_dim=embedding_matrix_init.shape[0],
        output_dim=embedding_matrix_init.shape[1],
        input_length=seq_max,
        weights=[embedding_matrix_init],
        trainable=True,
    )(text_input)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    x = SpatialDropout1D(0.15)(x)
    x = Bidirectional(GRU(40, return_sequences=True))(x)
    x = SpatialDropout1D(0.15)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(250, activation="linear")(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Dense(150, activation="linear")(x)
    x = Dropout(0.15)(x)
    x = LeakyReLU(alpha=0.05)(x)

    if nb_meta == 0:
        inputs = text_input
        concatenate_2 = x
    else:
        Meta_input = Input(shape=(nb_meta,), dtype="float32")
        inputs = [text_input, Meta_input]

        concatenate_1 = Meta_input
        y = Dense(150, activation="linear")(concatenate_1)
        y = Dropout(0.2)(y)
        y = LeakyReLU(alpha=0.05)(y)
        y = Dense(100, activation="linear")(y)
        y = Dropout(0.2)(y)
        y = LeakyReLU(alpha=0.05)(y)
        y = Dense(80, activation="linear")(y)
        y = Dropout(0.2)(y)
        y = LeakyReLU(alpha=0.05)(y)
        concatenate_2 = Concatenate(axis=1)([x, y])

    z = Dense(200, activation="linear")(concatenate_2)
    z = Dropout(0.2)(z)
    z = LeakyReLU(alpha=0.05)(z)
    z = Dense(100, activation="linear")(z)
    z = Dropout(0.2)(z)
    z = LeakyReLU(alpha=0.05)(z)
    output = Dense(ntargets, activation=activation)(z)

    model = Model(inputs=inputs, outputs=output)

    model.compile(optimizer=Adam(), loss=loss, metrics=["accuracy"])

    return model


def transformers_model(
    embedding_matrix_init,
    ntargets=18,
    seq_max=100,
    nb_meta=134,
    loss="categorical_crossentropy",
    activation="softmax",
):
    """Pre-defined architecture of a Transformer model.

    Parameters
    ----------
    embedding_matrix_init : np.array,
       Pretrained embedding matrix.

    ntargets : int, optional
       Dimension of model output.
       Default value, 18.

    seq_max : int, optional
       Maximum input length.
       Default value, 100.

    nb_meta : int, optional
       Dimension of meta data input.
       Default value, 252.

    loss : str, optional
       Loss function for training.
       Default value, 'categorical_crossentropy'.

    activation : str, optional
        Activation function.
        Default value, 'softmax'.

    Returns
    -------
    Model instance
    """

    text_input = Input(shape=(seq_max,), dtype="int32")

    x = Embedding(
        input_dim=embedding_matrix_init.shape[0],
        output_dim=embedding_matrix_init.shape[1],
        input_length=seq_max,
        weights=[embedding_matrix_init],
        trainable=False,
    )(text_input)

    x, mask = PositionalEncoding(position=seq_max, d_model=x.shape[2], pad_index=0)(
        inputs=x, seq=text_input
    )
    x = TransformerEncoderLayer(num_heads=10, d_model=x.shape[2], dff=x.shape[2])(
        x, mask
    )
    x = TransformerEncoderLayer(num_heads=10, d_model=x.shape[2], dff=x.shape[2])(
        x, None
    )
    x = Dense(150, activation="linear")(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = GlobalMaxPooling1D()(x)

    if nb_meta == 0:
        inputs = text_input
        concatenate_2 = x
    else:
        Meta_input = Input(shape=(nb_meta,), dtype="float32")
        inputs = [text_input, Meta_input]

        concatenate_1 = Meta_input
        y = Dense(150, activation="linear")(concatenate_1)
        y = Dropout(0.2)(y)
        y = LeakyReLU(alpha=0.05)(y)
        y = Dense(100, activation="linear")(y)
        y = Dropout(0.2)(y)
        y = LeakyReLU(alpha=0.05)(y)
        y = Dense(80, activation="linear")(y)
        y = Dropout(0.2)(y)
        y = LeakyReLU(alpha=0.05)(y)
        concatenate_2 = Concatenate(axis=1)([x, y])

    z = Dense(200, activation="linear")(concatenate_2)
    z = Dropout(0.2)(z)
    z = LeakyReLU(alpha=0.05)(z)
    z = Dense(100, activation="linear")(z)
    z = Dropout(0.2)(z)
    z = LeakyReLU(alpha=0.05)(z)
    output = Dense(ntargets, activation=activation)(z)

    model = Model(inputs=inputs, outputs=output)

    model.compile(optimizer=Adam(), loss=loss, metrics=["accuracy"])

    return model


def bert_model(
    ntargets=18,
    seq_max=100,
    nb_meta=134,
    loss="categorical_crossentropy",
    activation="softmax",
    bert_model="jplu/tf-camembert-base",
):
    """Pre-defined architecture of a pre-trained Bert model.

    Parameters
    ----------
    ntargets : int, optional
       Dimension of model output.
       Default value, 18.

    seq_max : int, optional
       Maximum input length.
       Default value, 100.

    nb_meta : int, optional
       Dimension of meta data input.
       Default value, 252.

    loss : str, optional
       Loss function for training.
       Default value, 'categorical_crossentropy'.

    activation : str, optional
        Activation function.
        Default value, 'softmax'.

    bert_model : str, optional
        Model name from HuggingFace library or path to local model
        Only Camembert and Flaubert supported
        Default value, 'camembert-base'

    Returns
    -------
    Model instance
    """
    # Prevent the HuggingFace dependency
    try:
        from transformers import TFCamembertModel, TFFlaubertModel
    except ModuleNotFoundError:
        raise (
            """Please install transformers 3.4.0 (only version currently supported)
            pip install melusine[transformers]"""
        )

    text_input = Input(shape=(seq_max,), dtype="int32")
    attention_input = Input(shape=(seq_max,), dtype="int32")
    if "camembert" in bert_model.lower():
        x = TFCamembertModel.from_pretrained(bert_model)(
            inputs=text_input, attention_mask=attention_input
        )[1]
    elif "flaubert" in bert_model.lower():
        x = TFFlaubertModel.from_pretrained(bert_model)(
            inputs=text_input, attention_mask=attention_input
        )[0][:, 0, :]
    else:
        raise NotImplementedError(
            "Bert model {} is not implemented.".format(bert_model)
        )

    if nb_meta == 0:
        inputs = [text_input, attention_input]
        concatenate_2 = x
    else:
        Meta_input = Input(shape=(nb_meta,), dtype="float32")
        inputs = [text_input, attention_input, Meta_input]

        concatenate_1 = Meta_input
        y = Dense(150, activation="linear")(concatenate_1)
        y = Dropout(0.2)(y)
        y = LeakyReLU(alpha=0.05)(y)
        y = Dense(100, activation="linear")(y)
        y = Dropout(0.2)(y)
        y = LeakyReLU(alpha=0.05)(y)
        y = Dense(80, activation="linear")(y)
        y = Dropout(0.2)(y)
        y = LeakyReLU(alpha=0.05)(y)
        concatenate_2 = Concatenate(axis=1)([x, y])

    z = Dense(200, activation="linear")(concatenate_2)
    z = Dropout(0.2)(z)
    z = LeakyReLU(alpha=0.05)(z)
    z = Dense(100, activation="linear")(z)
    z = Dropout(0.2)(z)
    z = LeakyReLU(alpha=0.05)(z)
    output = Dense(ntargets, activation=activation)(z)

    model = Model(inputs=inputs, outputs=output)

    model.compile(optimizer=Adam(learning_rate=5e-5), loss=loss, metrics=["accuracy"])

    return model
