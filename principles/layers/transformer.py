import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Layer, LayerNormalization, Dropout, Dense, Embedding


class Transformer(Layer):
    def __init__(self,
                 inputs_vocab_size,
                 target_vocab_size,
                 encoder_count,
                 decoder_count,
                 n_head,
                 d_model,
                 d_ff,
                 dropout_prob):
        """
        :param inputs_vocab_size: 输入序列词表大小
        :param target_vocab_size: 输出序列词表大小
        :param encoder_count: 编码器数量
        :param decoder_count: 解码器数量
        :param n_head: 注意力头数
        :param d_model: 多头注意力中所有向量维度之和：d_model / n_head = d_q = d_k = d_v
        :param d_ff: 前馈神经网络隐藏层维度
        :param dropout_prob: dropout概率

        Input shape
          - inputs: ``(batch_size, sor_len)``
          - target: ``(batch_size, tar_len)``
          - inputs_padding_mask: ``(batch_size, n_head, sor_len, sor_len)``
          - look_ahead_mask: ``(batch_size, n_head, tar_len, tar_len)``
          - target_padding_mask: ``(batch_size, n_head, tar_len, sor_len)``
        Output shape
          - output: ``(batch_size, tar_len, target_vocab_size)``
        """
        super(Transformer, self).__init__()

        # model hyper parameter variables
        self.encoder_count = encoder_count
        self.decoder_count = decoder_count
        self.encoder_embedding_layer = EmbeddingLayer(inputs_vocab_size, d_model)
        self.encoder_embedding_dropout = Dropout(dropout_prob)
        self.decoder_embedding_layer = EmbeddingLayer(target_vocab_size, d_model)
        self.decoder_embedding_dropout = Dropout(dropout_prob)

        self.encoder_layers = [
            EncoderLayer(n_head, d_model, d_ff, dropout_prob) for _ in range(encoder_count)
        ]
        self.decoder_layers = [
            DecoderLayer(n_head, d_model, d_ff, dropout_prob) for _ in range(decoder_count)
        ]
        self.linear = Dense(target_vocab_size)

    def call(self, inputs, training=None, **kwargs):
        target = inputs['target']
        inputs_padding_mask = inputs['inputs_padding_mask']
        look_ahead_mask = inputs['look_ahead_mask']
        target_padding_mask = inputs['target_padding_mask']
        inputs = inputs['inputs']
        encoder_tensor = self.encoder_embedding_layer(inputs)
        encoder_tensor = self.encoder_embedding_dropout(encoder_tensor, training=training)

        for i in range(self.encoder_count):
            encoder_tensor, _ = self.encoder_layers[i](encoder_tensor, mask=inputs_padding_mask, training=training)
        target = self.decoder_embedding_layer(target)
        decoder_tensor = self.decoder_embedding_dropout(target, training=training)
        for i in range(self.decoder_count):
            decoder_tensor, _, _ = self.decoder_layers[i](
                {'decoder_inputs': decoder_tensor,
                 'encoder_output': encoder_tensor,
                 'look_ahead_mask': look_ahead_mask,
                 'padding_mask': target_padding_mask},
                training=training
            )
        return self.linear(decoder_tensor)


def positional_encoding(max_len, emb_dim):
    """
      位置编码
    """
    pos = np.expand_dims(np.arange(0, max_len), axis=1)
    index = np.expand_dims(np.arange(0, emb_dim), axis=0)
    pe = pos / np.power(10000, (index - index % 2) / np.float32(emb_dim))
    pe[:, 0::2] = np.sin(pe[:, 0::2])
    pe[:, 1::2] = np.cos(pe[:, 1::2])
    return pe


class EmbeddingLayer(Layer):
    def __init__(self, vocab_size, d_model):
        """
        inputs与outputs的Embedding层
        :param vocab_size: 词表词数
        :param d_model: 多头注意力中所有向量维度之和：d_model / n_head = d_q = d_k = d_v

        Input shape
          - inputs: ``(batch_size, seq_len)``
        Output shape
          - output: ``(batch_size, seq_len, d_model)``
        """
        super(EmbeddingLayer, self).__init__()
        self.d_model = d_model
        self.embedding = Embedding(vocab_size, d_model)

    def call(self, inputs, **kwargs):
        max_sequence_len = inputs.shape[1]
        # 对嵌入向量进行缩放，使其与位置编码大小匹配
        output = self.embedding(inputs) * tf.sqrt(tf.cast(self.d_model, dtype=tf.float32))
        output += positional_encoding(max_sequence_len, self.d_model)
        return output


class EncoderLayer(Layer):
    def __init__(self, n_head, d_model, d_ff, dropout_prob):
        """
        单个Encoder层

        Input shape
          - inputs: ``(batch_size, sor_len, d_model)``
          - mask:  ``(batch_size, n_head, sor_len, sor_len)``
        Output shape
          - output: ``(batch_size, sor_len, d_model)``
          - attention_weight: ``(batch_size, n_head, sor_len, sor_len)``
        """
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(n_head, d_model)
        self.dropout_1 = Dropout(dropout_prob)
        self.layer_norm_1 = LayerNormalization(epsilon=1e-6)

        self.position_wise_feed_forward_layer = PositionWiseFeedForwardLayer(d_ff, d_model)
        self.dropout_2 = Dropout(dropout_prob)
        self.layer_norm_2 = LayerNormalization(epsilon=1e-6)

    def call(self, inputs, mask=None, training=None, **kwargs):
        output, attention = self.multi_head_attention({'query': inputs, 'key': inputs, 'value': inputs}, mask=mask)
        output = self.dropout_1(output, training=training)
        output = self.layer_norm_1(tf.add(inputs, output))  # residual network
        attention_output = output

        output = self.position_wise_feed_forward_layer(output)
        output = self.dropout_2(output, training=training)
        output = self.layer_norm_2(tf.add(attention_output, output))  # residual network
        return output, attention


class DecoderLayer(Layer):
    def __init__(self, n_head, d_model, d_ff, dropout_prob):
        """
        单个Decoder层

        Input shape
          - decoder_inputs:  ``(batch_size, tar_len, d_model)``
          - encoder_output:  ``(batch_size, sor_len, d_model)``
          - look_ahead_mask: ``(batch_size, n_head, tar_len, tar_len)``
          - padding_mask:    ``(batch_size, n_head, tar_len, sor_len)``
        Output shape
          - output: ``(batch_size, tar_len, d_model)``
          - attention_1: ``(batch_size, n_head, tar_len, tar_len)``
          - attention_2: ``(batch_size, n_head, tar_len, sor_len)``
        """
        super(DecoderLayer, self).__init__()
        self.masked_multi_head_attention = MultiHeadAttention(n_head, d_model)
        self.dropout_1 = Dropout(dropout_prob)
        self.layer_norm_1 = LayerNormalization(epsilon=1e-6)

        self.encoder_decoder_attention = MultiHeadAttention(n_head, d_model)
        self.dropout_2 = Dropout(dropout_prob)
        self.layer_norm_2 = LayerNormalization(epsilon=1e-6)

        self.position_wise_feed_forward_layer = PositionWiseFeedForwardLayer(d_ff, d_model)
        self.dropout_3 = Dropout(dropout_prob)
        self.layer_norm_3 = LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=None, **kwargs):
        decoder_inputs = inputs['decoder_inputs']
        encoder_output = inputs['encoder_output']
        look_ahead_mask = inputs['look_ahead_mask']
        padding_mask = inputs['padding_mask']
        output, attention_1 = self.masked_multi_head_attention(
            {'query': decoder_inputs, 'key': decoder_inputs, 'value': decoder_inputs},
            mask=look_ahead_mask
        )  # output：(batch_size, tar_len, d_model)
        output = self.dropout_1(output, training=training)
        query = self.layer_norm_1(tf.add(decoder_inputs, output))  # residual network
        output, attention_2 = self.encoder_decoder_attention(
            {'query': query, 'key': encoder_output, 'value': encoder_output},
            mask=padding_mask
        )  # output：(batch_size, tar_len, d_model)
        output = self.dropout_2(output, training=training)
        encoder_decoder_attention_output = self.layer_norm_2(tf.add(output, query))

        output = self.position_wise_feed_forward_layer(encoder_decoder_attention_output)
        output = self.dropout_3(output, training=training)
        output = self.layer_norm_3(tf.add(encoder_decoder_attention_output, output))  # residual network
        return output, attention_1, attention_2


class PositionWiseFeedForwardLayer(Layer):
    """
    用于每个Encoder和Decoder中最后的前馈神经网络，包含两个线性变换，并在两个变换中间有一个ReLU激活函数
    """
    def __init__(self, d_ff=2048, d_model=512):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.w_1 = Dense(d_ff)  # 第一层输入512维,输出2048维
        self.w_2 = Dense(d_model)  # 第二层输入2048维，输出512维

    def call(self, inputs, **kwargs):
        inputs = self.w_1(inputs)
        inputs = tf.nn.relu(inputs)
        return self.w_2(inputs)


class MultiHeadAttention(Layer):
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
        super(MultiHeadAttention, self).__init__()

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

        output, attention = self.scaled_dot_product({'query': query, 'key': key, 'value': value}, mask=mask)  # output shape: (batch_size, n_head, tar_len, d_k)
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
