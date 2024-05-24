import tensorflow as tf
import numpy as np
from principles.layers.transformer import ScaledDotProductAttention, MultiHeadAttention, PositionWiseFeedForwardLayer, EncoderLayer, DecoderLayer, positional_encoding, EmbeddingLayer, Transformer
from tensorflow.python.keras.layers import Attention


class LayersTest(tf.test.TestCase):
    def __init__(self, methodName='LayersTest'):
        super(LayersTest, self).__init__(methodName=methodName)

    def test_self_attention(self):
        batch_size = 3
        d_k = 16
        d_v = 32
        sor_len = 10
        tar_len = 15
        query = tf.constant(np.random.rand(batch_size, tar_len, d_k))
        key = tf.constant(np.random.rand(batch_size, sor_len, d_k))
        value = tf.constant(np.random.rand(batch_size, sor_len, d_v))
        mask = tf.constant(np.random.rand(batch_size, tar_len, sor_len), dtype=tf.float32)
        layer = ScaledDotProductAttention(d_k)
        output, attention_weight = layer({'query': query, 'key': key, 'value': value}, mask=mask)
        output_shape = np.ones([batch_size, tar_len, d_v])
        weight_shape = np.ones([batch_size, tar_len, sor_len])
        self.assertShapeEqual(output_shape, output)
        self.assertShapeEqual(weight_shape, attention_weight)

    def test_self_attention2(self):
        batch_size = 3
        n_head = 4
        d_k = 16
        sor_len = 10
        tar_len = 15
        query = tf.constant(np.random.rand(batch_size, n_head, tar_len, d_k))
        key = tf.constant(np.random.rand(batch_size, n_head, sor_len, d_k))
        value = tf.constant(np.random.rand(batch_size, n_head, sor_len, d_k))
        layer = ScaledDotProductAttention(d_k)
        output, attention_weight = layer({'query': query, 'key': key, 'value': value})
        output_shape = np.ones([batch_size, n_head, tar_len, d_k])
        weight_shape = np.ones([batch_size, n_head, tar_len, sor_len])
        self.assertShapeEqual(output_shape, output)
        self.assertShapeEqual(weight_shape, attention_weight)

    def test_keras_attention(self):
        batch_size = 3
        d_k = 16
        d_v = 32
        sor_len = 10
        tar_len = 15
        query = tf.constant(np.random.rand(batch_size, tar_len, d_k))
        key = tf.constant(np.random.rand(batch_size, sor_len, d_k))
        value = tf.constant(np.random.rand(batch_size, sor_len, d_v))
        layer = Attention(use_scale=True)
        output, attention_weight = layer([query, value, key], return_attention_scores=True)
        print(output)
        print(attention_weight)
        output_shape = np.ones([batch_size, tar_len, d_v])
        weight_shape = np.ones([batch_size, tar_len, sor_len])
        self.assertShapeEqual(output_shape, output)
        self.assertShapeEqual(weight_shape, attention_weight)

    def test_multi_head_attention(self):
        batch_size = 3
        sor_len = 10
        tar_len = 15
        d_model = 32
        n_head = 4
        layer = MultiHeadAttention(n_head=n_head, d_model=d_model)
        query = tf.constant(np.random.rand(batch_size, tar_len, d_model))
        key = tf.constant(np.random.rand(batch_size, sor_len, d_model))
        value = tf.constant(np.random.rand(batch_size, sor_len, d_model))
        mask = tf.constant(np.random.rand(batch_size, n_head, tar_len, sor_len), dtype=tf.float32)
        output, attention = layer({'query': query, 'key': key, 'value': value}, mask=mask)
        output_shape = np.ones([batch_size, tar_len, d_model])
        attention_shape = np.ones([batch_size, n_head, tar_len, sor_len])
        self.assertShapeEqual(output_shape, output)
        self.assertShapeEqual(attention_shape, attention)

    def test_feed_forward_layer(self):
        batch_size = 3
        tar_len = 15
        d_ff = 2048
        d_model = 512
        layer = PositionWiseFeedForwardLayer(d_ff=d_ff, d_model=d_model)
        tensor = tf.constant(np.random.rand(batch_size, tar_len, d_model))
        output = layer(tensor)
        output_shape = np.ones([batch_size, tar_len, d_model])
        self.assertShapeEqual(output_shape, output)

    def test_encoder_layer(self):
        batch_size = 3
        sor_len = 10
        n_head = 4
        d_model = 32
        layer = EncoderLayer(n_head=n_head, d_model=d_model, d_ff=1024, dropout_prob=0.1)
        inputs = tf.constant(np.random.rand(batch_size, sor_len, d_model))
        mask = tf.constant(np.random.rand(batch_size, n_head, sor_len, sor_len), dtype=tf.float32)
        output, attention = layer(inputs, mask=mask)
        output_shape = np.ones([batch_size, sor_len, d_model])
        self.assertShapeEqual(output_shape, output)
        attention_shape = np.ones([batch_size, n_head, sor_len, sor_len])
        self.assertShapeEqual(attention_shape, attention)

    def test_decoder_layer(self):
        batch_size = 3
        sor_len = 10
        tar_len = 15
        n_head = 4
        d_model = 512
        layer = DecoderLayer(n_head=n_head, d_model=d_model, d_ff=1024, dropout_prob=0.1)
        decoder_inputs = tf.constant(np.random.rand(batch_size, tar_len, d_model))
        encoder_output = tf.constant(np.random.rand(batch_size, sor_len, d_model))
        look_ahead_mask = tf.constant(np.random.rand(batch_size, n_head, tar_len, tar_len))
        padding_mask = tf.constant(np.random.rand(batch_size, n_head, tar_len, sor_len))
        output, attention_1, attention_2 = layer({'decoder_inputs': decoder_inputs, 'encoder_output': encoder_output,
                                                  'look_ahead_mask': look_ahead_mask, 'padding_mask': padding_mask})
        output_shape = np.ones([batch_size, tar_len, d_model])
        self.assertShapeEqual(output_shape, output)
        attention_1_shape = np.ones([batch_size, n_head, tar_len, tar_len])
        self.assertShapeEqual(attention_1_shape, attention_1)
        attention_2_shape = np.ones([batch_size, n_head, tar_len, sor_len])
        self.assertShapeEqual(attention_2_shape, attention_2)

    def test_embedding_layer(self):
        batch_size = 3
        vocab_size = 100
        seq_len = 15
        d_model = 64
        layer = EmbeddingLayer(vocab_size=vocab_size, d_model=d_model)
        inputs = tf.constant(np.random.rand(batch_size, seq_len))
        output = layer(inputs)
        output_shape = np.ones([batch_size, seq_len, d_model])
        self.assertShapeEqual(output_shape, output)

    def test_transformer(self):
        batch_size = 3
        sor_len = 10
        tar_len = 15
        inputs_vocab_size = 100
        target_vocab_size = 150
        encoder_count = 5
        decoder_count = 6
        n_head = 4
        d_model = 64
        layer = Transformer(inputs_vocab_size=inputs_vocab_size, target_vocab_size=target_vocab_size,
                            encoder_count=encoder_count, decoder_count=decoder_count, n_head=n_head, d_model=d_model,
                            d_ff=1024, dropout_prob=0.1)
        inputs = tf.constant(np.random.rand(batch_size, sor_len))
        target = tf.constant(np.random.rand(batch_size, tar_len))
        inputs_padding_mask = tf.constant(np.random.rand(batch_size, 1, sor_len, sor_len), dtype=tf.float32)
        look_ahead_mask = tf.constant(np.random.rand(batch_size, 1, tar_len, tar_len), dtype=tf.float32)
        target_padding_mask = tf.constant(np.random.rand(batch_size, 1, tar_len, sor_len), dtype=tf.float32)
        outputs = layer({'inputs': inputs, 'target': target, 'inputs_padding_mask': inputs_padding_mask, 'look_ahead_mask': look_ahead_mask, 'target_padding_mask': target_padding_mask})
        outputs_shape = np.ones([batch_size, tar_len, target_vocab_size])
        self.assertShapeEqual(outputs_shape, outputs)

    def test_positional_encoding(self):
        pe = positional_encoding(50, 128)
        import matplotlib.pyplot as plt
        plt.imshow(pe)
        plt.colorbar()
        plt.show()

        mat = np.matmul(pe, pe.T)
        plt.imshow(mat)
        plt.colorbar()
        plt.show()
