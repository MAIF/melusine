from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.layers import GlobalMaxPooling1D


def default_meta_archi(nb_meta):
    meta_input = Input(shape=(nb_meta,), dtype="float32")

    meta_net = Dense(150, activation="linear")(meta_input)
    meta_net = Dropout(0.2)(meta_net)
    meta_net = LeakyReLU(alpha=0.05)(meta_net)
    meta_net = Dense(100, activation="linear")(meta_net)
    meta_net = Dropout(0.2)(meta_net)
    meta_net = LeakyReLU(alpha=0.05)(meta_net)
    meta_net = Dense(80, activation="linear")(meta_net)
    meta_net = Dropout(0.2)(meta_net)
    meta_net = LeakyReLU(alpha=0.05)(meta_net)

    return meta_input, meta_net


def default_dense_archi(input_net):
    dense_net = Dense(200, activation="linear")(input_net)
    dense_net = Dropout(0.2)(dense_net)
    dense_net = LeakyReLU(alpha=0.05)(dense_net)
    dense_net = Dense(100, activation="linear")(dense_net)
    dense_net = Dropout(0.2)(dense_net)
    dense_net = LeakyReLU(alpha=0.05)(dense_net)

    return dense_net


def default_cnn_archi(input_net):
    cnn_net = Conv1D(200, 2, padding="same", activation="linear", strides=1)(input_net)
    cnn_net = SpatialDropout1D(0.15)(cnn_net)
    cnn_net = BatchNormalization()(cnn_net)
    cnn_net = LeakyReLU(alpha=0.05)(cnn_net)
    cnn_net = Conv1D(250, 2, padding="same", activation="linear", strides=1)(cnn_net)
    cnn_net = SpatialDropout1D(0.15)(cnn_net)
    cnn_net = LeakyReLU(alpha=0.05)(cnn_net)
    cnn_net = Dropout(0.15)(cnn_net)
    cnn_net = GlobalMaxPooling1D()(cnn_net)
    cnn_net = Dense(250, activation="linear")(cnn_net)
    cnn_net = LeakyReLU(alpha=0.05)(cnn_net)
    cnn_net = Dense(150, activation="linear")(cnn_net)
    cnn_net = Dropout(0.15)(cnn_net)
    cnn_net = LeakyReLU(alpha=0.05)(cnn_net)

    return cnn_net


def default_multiclass_archi(input_net, n_targets):
    output = Dense(n_targets, activation="softmax")(input_net)
    return output
