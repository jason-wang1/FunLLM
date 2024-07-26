import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization, Dropout, Dense, Embedding


class Qwen2DecoderLayer(Layer):
    def __init__(self, n_head, d_model, hidden_size, intermediate_size):
        super(Qwen2DecoderLayer, self).__init__()
        self.self_attention = Qwen2Attention(n_head, d_model)
        self.norm = LayerNormalization(epsilon=1e-6)
        self.encoder_attention = Qwen2Attention(n_head, d_model)
        self.mlp = Qwen2MLP(hidden_size, intermediate_size)

    def call(self, inputs, mask=None, training=None, **kwargs):
        residual = inputs
        hidden_states = inputs
        hidden_states = self.norm(hidden_states)
        hidden_states, self_attn_weights = self.self_attention({'query': hidden_states, 'key': hidden_states, 'value': hidden_states})
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states


class Qwen2MLP(Layer):
    def __init__(self, hidden_size, intermediate_size):
        """
        Input shape
            - tensor: ``(batch_size, seq_len, hidden_size)``

        Output shape
            - output: ``(batch_size, seq_len, hidden_size)``
        """
        super(Qwen2MLP, self).__init__()
        self.gate_proj = Dense(intermediate_size, use_bias=False)
        self.up_proj = Dense(intermediate_size, use_bias=False)
        self.down_proj = Dense(hidden_size, use_bias=False)

    def call(self, inputs, *args, **kwargs):
        down_proj = self.down_proj(tf.nn.relu(self.gate_proj(inputs)) * self.up_proj(inputs))
        return down_proj


class Qwen2Attention(Layer):
    def __init__(self, n_head, d_model):
        """
          多头注意力机制

          param n_head:  注意力平行头数量
          param d_model: 多头注意力中所有向量维度之和：d_model / n_head = d_q = d_k = d_v

          Input shape
            - query: ``(batch_size, tar_len, d_model)``
            - key:   ``(batch_size, sor_len, d_model)``
            - value: ``(batch_size, sor_len, d_model)``
            - mask:  ``(batch_size, n_head, tar_len, sor_len)``

          Output shape
            - output: ``(batch_size, tar_len, d_model)``
            - attention_weight: ``(batch_size, n_head, tar_len, sor_len)``
        """
        super(Qwen2Attention, self).__init__()

        # model hyper parameter variables
        self.n_head = n_head
        self.d_model = d_model

        if d_model % n_head != 0:
            raise ValueError(
                "d_model({}) % attention_head_count({}) is not zero.d_model must be multiple of attention_head_count.".format(
                    d_model, n_head
                )
            )

        self.d_k = d_model // n_head

        self.w_query = Dense(d_model)
        self.w_key = Dense(d_model)
        self.w_value = Dense(d_model)

        self.scaled_dot_product = ScaledDotProductAttention(self.d_k)

        self.dense = Dense(d_model)

    def call(self, inputs, mask=None, **kwargs):
        query = inputs['query']
        key = inputs['key']
        value = inputs['value']

        batch_size = tf.shape(query)[0]

        query = self.w_query(query)  # (batch_size, tar_len, d_model)
        key = self.w_key(key)  # (batch_size, sor_len, d_model)
        value = self.w_value(value)  # (batch_size, sor_len, d_model)

        query = self.split_head(query, batch_size)  # (batch_size, n_head, tar_len, d_k)
        key = self.split_head(key, batch_size)  # (batch_size, n_head, sor_len, d_k)
        value = self.split_head(value, batch_size)  # (batch_size, n_head, sor_len, d_k)

        output, attention = self.scaled_dot_product({'query': query, 'key': key, 'value': value},
                                                    mask=mask)  # output shape: (batch_size, n_head, tar_len, d_k)
        output = self.concat_head(output, batch_size)  # (batch_size, tar_len, d_model)
        return self.dense(output), attention

    def split_head(self, tensor, batch_size):
        """
        将张量沿着最后一个维度分成num_heads个头
        Input shape
            - tensor: ``(batch_size, seq_len, d_model)``

        Output shape
            - output: ``(batch_size, n_head, seq_len, d_k)``
        """
        return tf.transpose(
            tf.reshape(
                tensor,
                (batch_size, -1, self.n_head, self.d_k)
                # tensor: (batch_size, seq_len, n_head, d_k)
            ),
            [0, 2, 1, 3]
        )

    def concat_head(self, tensor, batch_size):
        """
        将多个头合并回单个张量
        Input shape
            - tensor: ``(batch_size, n_head, seq_len, d_k)``

        Output shape
            - output: ``(batch_size, seq_len, d_model)``
        """
        return tf.reshape(
            tf.transpose(tensor, [0, 2, 1, 3]),
            (batch_size, -1, self.n_head * self.d_k)
        )


class ScaledDotProductAttention(Layer):
    def __init__(self, d_k):
        """
          注意力权重

          param d_k: key的维度

          Input shape
            - query: ``(batch_size, tar_len, d_k)``
            - key:   ``(batch_size, sor_len, d_k)``
            - value: ``(batch_size, sor_len, d_v)``
            - mask:  ``(batch_size, tar_len, sor_len)`` 当mask=1时，权重将置为0.0

          Output shape
            - output: ``(batch_size, tar_len, d_v)``
            - attention_weight: ``(batch_size, tar_len, sor_len)``
        """
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def call(self, inputs, mask=None, **kwargs):
        query = inputs['query']
        key = inputs['key']
        value = inputs['value']
        score = tf.matmul(query, key, transpose_b=True) / self.d_k  # (batch_size, tar_len, sor_len)
        if mask is not None:
            score += (mask * -1.e9)
        attention_weight = tf.nn.softmax(score, axis=-1)
        output = tf.matmul(attention_weight, value)  # (batch_size, tar_len, d_v)
        return output, attention_weight
