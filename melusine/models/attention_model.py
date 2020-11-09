# Adapted from https://www.tensorflow.org/tutorials/text/transformer
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):

        self.num_heads = num_heads
        self.d_model = d_model
        super(MultiHeadAttention, self).__init__(**kwargs)
        assert d_model % self.num_heads == 0

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def get_config(self):
        config = {
            "num_heads": self.num_heads,
            "d_model": self.d_model,
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.d_model // self.num_heads)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        (
            scaled_attention,
            attention_weights,
        ) = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

    @staticmethod
    def scaled_dot_product_attention(q, k, v, mask):
        """Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead)
        but it must be broadcastable for addition.

        Args:
          q: query shape == (..., seq_len_q, depth)
          k: key shape == (..., seq_len_k, depth)
          v: value shape == (..., seq_len_v, depth_v)
          mask: Float tensor with shape broadcastable
                to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
          output, attention_weights
        """

        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += mask * -1e9

            # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(
            scaled_attention_logits, axis=-1
        )  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights


class PositionalEncoding(layers.Layer):
    def __init__(self, position, d_model, pad_index, **kwargs):
        super(PositionalEncoding, self).__init__()
        self.position = int(position)
        self.d_model = int(d_model)
        self.positional_encoding = self.get_positional_encoding()
        self.pad_index = pad_index

    def get_config(self):
        config = {
            "position": self.position,
            "pad_index": self.pad_index,
            "d_model": self.d_model,
        }
        base_config = super(PositionalEncoding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_positional_encoding(self):
        angle_rads = self.get_angles(
            np.arange(self.position)[:, np.newaxis],
            np.arange(self.d_model)[np.newaxis, :],
            self.d_model,
        )

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, self.pad_index), tf.float32)

        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    def call(self, inputs, seq):
        mask = self.create_padding_mask(seq)

        seq_len = tf.shape(inputs)[1]
        inputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        inputs += self.positional_encoding[:, :seq_len, :]
        return inputs, mask

    @staticmethod
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model=100, num_heads=10, dff=100, rate=0.1, **kwargs):
        self.d_model = int(d_model)
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        super(TransformerEncoderLayer, self).__init__(**kwargs)

        self.mha = MultiHeadAttention(self.d_model, self.num_heads)
        self.dense1 = tf.keras.layers.Dense(self.dff, activation="relu")
        self.dense2 = tf.keras.layers.Dense(self.d_model)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(self.rate)
        self.dropout2 = tf.keras.layers.Dropout(self.rate)

    def get_config(self):
        config = {
            "d_model": self.d_model,
            "dff": self.dff,
            "num_heads": self.num_heads,
            "rate": self.rate,
        }
        base_config = super(TransformerEncoderLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        out1 = self.dense1(out1)
        ffn_output = self.dense2(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(
            out1 + ffn_output
        )  # (batch_size, input_seq_len, d_model)

        return out2
