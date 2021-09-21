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
from tensorflow.keras.models import Sequential
import tensorflow as tf

try:
    from tensorflow_probability import layers as tfpl
    from tensorflow_probability import distributions as tfd
except ModuleNotFoundError:
    raise (
        """Please install tensorflow-probability 0.14.0
        pip install melusine[tf_probability]"""
    )


def negative_log_likelihood(y_true, y_pred):
    """Negative log likelihood."""
    return -y_pred.log_prob(y_true)


def bayesian_cnn_with_meta_model(
    embedding_matrix_init,
    ntargets,
    seq_max,
    nb_meta,
    loss=None, 
    activation="softmax",):
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
    
    z = Dense(units=tfpl.OneHotCategorical.params_size(ntargets))(z)
    outputs = tfpl.OneHotCategorical(ntargets, convert_to_tensor_fn=tfd.Distribution.mode)(z)
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=Adam(), loss=lambda y, p_y: -p_y.log_prob(y), metrics=["accuracy"])
    return model

def spike_and_slab(event_shape, dtype):
    "Define the spike and slab distribution."
    distribution = tfd.Mixture(
        cat=tfd.Categorical(probs=[0.5, 0.5]),
        components=[
            tfd.Independent(tfd.Normal(
                loc=tf.zeros(event_shape, dtype=dtype), 
                scale=1.0*tf.ones(event_shape, dtype=dtype)),
                            reinterpreted_batch_ndims=1), # TODO scale : 1.0*tf.ones
            tfd.Independent(tfd.Normal(
                loc=tf.zeros(event_shape, dtype=dtype), 
                scale=10.0*tf.ones(event_shape, dtype=dtype)), 
                            reinterpreted_batch_ndims=1)],  # TODO scale : 10.0*tf.ones
    name='spike_and_slab')
    return distribution


def prior(kernel_size, bias_size, dtype = None):
    """
    Multivariate Normal Diagonal non-trainable prior (loc=0, scale=1).
    """
    n = kernel_size + bias_size
    prior_model = Sequential([
        tfpl.DistributionLambda(
            lambda t: tfd.MultivariateNormalDiag(
                loc=tf.zeros(n), scale_diag=1.5*tf.ones(n)
            )
        )
    ])
    return prior_model


def posterior(kernel_size, bias_size, dtype=None):
    """
    Posterior returning an Independent Normal distribution.
    """
    n = kernel_size + bias_size
    return Sequential([
        tfpl.VariableLayer(tfpl.IndependentNormal.params_size(n), dtype=dtype),
        tfpl.IndependentNormal(n)
    ])

def variational_cnn_model(
    embedding_matrix_init,
    ntargets,
    seq_max,
    nb_meta,
    loss=None, 
    activation="softmax",
):
    """Pre-defined architecture of a CNN model.

    Parameters
    ----------
    TODO DEFINE

    Returns
    -------
    Model instance
    """
    N = 40 ################################################ TODO replace 40 en len(X)
    divergence_fn = lambda q, p, _ : tfd.kl_divergence(q, p) / N
    text_input = Input(shape=(seq_max,), dtype="int32")

    x = Embedding(
        input_dim=embedding_matrix_init.shape[0],
        output_dim=embedding_matrix_init.shape[1],
        input_length=seq_max,
        weights=[embedding_matrix_init],
        trainable=False,
    )(text_input)
    x = tfpl.Convolution1DReparameterization(
        filters=2**7, 
        kernel_size=3, 
        activation='relu',
        padding='valid', 
        kernel_prior_fn=tfpl.default_multivariate_normal_fn,
        kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
        kernel_divergence_fn=divergence_fn,
        bias_prior_fn=tfpl.default_multivariate_normal_fn,
        bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
        bias_divergence_fn=divergence_fn
    )(x)
    x = GlobalMaxPooling1D()(x)
    inputs = [text_input]

    x = tfpl.DenseVariational(
        units=2**6,
        make_prior_fn=prior,
        make_posterior_fn=posterior,
        kl_weight=1/N
    )(x)
    
    x = tfpl.DenseVariational(
        units=tfpl.OneHotCategorical.params_size(ntargets),
        make_prior_fn=prior,
        make_posterior_fn=posterior,
        kl_weight=1/N
    )(x)

    outputs = tfpl.OneHotCategorical(ntargets, convert_to_tensor_fn=tfd.Distribution.mode)(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=Adam(), loss=lambda y, p_y: -p_y.log_prob(y), metrics=["accuracy"])
    return model

def flipout_cnn_model(
    embedding_matrix_init,
    ntargets,
    seq_max,
    nb_meta,
    loss=None, 
    activation="softmax",
):
    """Pre-defined architecture of a CNN model.

    Parameters
    ----------
    TODO DEFINE

    Returns
    -------
    Model instance
    """
    N = 40 ################################################ TODO replace 40 en len(X)
    divergence_fn = lambda q, p, _ : tfd.kl_divergence(q, p) / N
    text_input = Input(shape=(seq_max,), dtype="int32")

    x = Embedding(
        input_dim=embedding_matrix_init.shape[0],
        output_dim=embedding_matrix_init.shape[1],
        input_length=seq_max,
        weights=[embedding_matrix_init],
        trainable=False,
    )(text_input)
    x = tfpl.Convolution1DFlipout(
         filters=2**7, kernel_size=3, padding='SAME',
         kernel_divergence_fn=divergence_fn,
         activation=tf.nn.relu)(x)
    x = GlobalMaxPooling1D()(x)
    inputs = [text_input]
    x = tfpl.DenseFlipout(
         2**6, kernel_divergence_fn=divergence_fn,
         activation=tf.nn.relu)(x)
    x = tfpl.DenseFlipout(
        ntargets, kernel_divergence_fn=divergence_fn,
        activation=tf.nn.softmax)(x)

    outputs = tfpl.OneHotCategorical(ntargets, convert_to_tensor_fn=tfd.Distribution.mode)(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=Adam(), loss=lambda y, p_y: -p_y.log_prob(y), metrics=["accuracy"])
    return model
