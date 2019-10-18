import keras
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import Input
from keras.layers import concatenate
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import Dropout
from keras.layers import SpatialDropout1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import GRU
from keras.layers import Bidirectional


def cnn_model(embedding_matrix_init,
              ntargets,
              seq_max,
              nb_meta,
              loss):
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

    Returns
    -------
    Model instance
    """

    text_input = Input(shape=(seq_max,), dtype='int32')

    x = keras.layers.Embedding(input_dim=embedding_matrix_init.shape[0],
                               output_dim=embedding_matrix_init.shape[1],
                               input_length=seq_max,
                               weights=[embedding_matrix_init],
                               trainable=True)(text_input)
    x = Conv1D(200, 2, padding='same', activation='linear', strides=1)(x)
    x = SpatialDropout1D(0.15)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Conv1D(250, 2, padding='same', activation='linear', strides=1)(x)
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
        Meta_input = Input(shape=(nb_meta,), dtype='float32')
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
        concatenate_2 = concatenate([x, y])

    z = Dense(200, activation="linear")(concatenate_2)
    z = Dropout(0.2)(z)
    z = LeakyReLU(alpha=0.05)(z)
    z = Dense(100, activation="linear")(z)
    z = Dropout(0.2)(z)
    z = LeakyReLU(alpha=0.05)(z)
    outputs = Dense(ntargets, activation='softmax')(z)

    model = Model(inputs=inputs,
                  outputs=outputs)

    model.compile(optimizer=Adam(),
                  loss=loss,
                  metrics=['accuracy'])

    return model


def rnn_model(embedding_matrix_init,
              ntargets=18,
              seq_max=100,
              nb_meta=252,
              loss='categorical_crossentropy'):
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

    Returns
    -------
    Model instance
    """
    text_input = Input(shape=(seq_max,), dtype='int32')

    x = keras.layers.Embedding(input_dim=embedding_matrix_init.shape[0],
                               output_dim=embedding_matrix_init.shape[1],
                               input_length=seq_max,
                               weights=[embedding_matrix_init],
                               trainable=True)(text_input)
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
        Meta_input = Input(shape=(nb_meta,), dtype='float32')
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
        concatenate_2 = concatenate([x, y])

    z = Dense(200, activation="linear")(concatenate_2)
    z = Dropout(0.2)(z)
    z = LeakyReLU(alpha=0.05)(z)
    z = Dense(100, activation="linear")(z)
    z = Dropout(0.2)(z)
    z = LeakyReLU(alpha=0.05)(z)
    output = Dense(ntargets, activation='softmax')(z)

    model = Model(inputs=inputs,
                  outputs=output)

    model.compile(optimizer=Adam(),
                  loss=loss,
                  metrics=['accuracy'])

    return model
