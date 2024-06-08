import tensorflow as tf
from keras.layers import (Layer,
                          Dropout,
                          Lambda,
                          Add,
                          Activation,
                          Dense,
                          TimeDistributed,
                          Concatenate,
                          Conv1D,
                          LayerNormalization)
import keras.backend as k_back


# File from https://github.com/lsdefine/attention-is-all-you-need-keras.

class ScaledDotProductAttention(Layer):
    def __init__(self, attn_dropout=0.1, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.dropout = Dropout(attn_dropout)

    def call(self, q, k, v, mask):  # mask_k or mask_qk
        temper = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype='float32'))
        attn = Lambda(lambda x: k_back.batch_dot(x[0], x[1], axes=[2, 2]) / x[2])([q, k, temper])  # shape=(batch, q, k)
        if mask is not None:
            m_mask = Lambda(lambda x: (-1e+9) * (1. - k_back.cast(x, 'float32')))(mask)
            attn = Add()([attn, m_mask])
        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x: k_back.batch_dot(x[0], x[1]))([attn, v])
        return output, attn

    def get_config(self):
        config = super(ScaledDotProductAttention, self).get_config()
        config.update({
            'attn_dropout': self.dropout.rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MultiHeadAttention(Layer):
    # mode 0 - big matrices, faster; mode 1 - more clear implementation
    def __init__(self, n_head, d_model, dropout, mode=0, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.mode = mode
        self.n_head = n_head
        self.d_k = self.d_v = d_k = d_v = d_model // n_head
        self.dropout = dropout
        if mode == 0:
            self.qs_layer = Dense(n_head * d_k, use_bias=False)
            self.ks_layer = Dense(n_head * d_k, use_bias=False)
            self.vs_layer = Dense(n_head * d_v, use_bias=False)
        elif mode == 1:
            self.qs_layers = []
            self.ks_layers = []
            self.vs_layers = []
            for _ in range(n_head):
                self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
        self.attention = ScaledDotProductAttention()
        self.w_o = TimeDistributed(Dense(d_model))

    def call(self, q, k, v, mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head
        head, attn = None, None

        if self.mode == 0:
            qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
            ks = self.ks_layer(k)
            vs = self.vs_layer(v)

            def reshape1(x):
                s = tf.shape(x)  # [batch_size, len_q, n_head * d_k]
                x = tf.reshape(x, [s[0], s[1], n_head, s[2] // n_head])
                x = tf.transpose(x, [2, 0, 1, 3])
                x = tf.reshape(x, [-1, s[1], s[2] // n_head])  # [n_head * batch_size, len_q, d_k]
                return x

            qs = Lambda(reshape1)(qs)
            ks = Lambda(reshape1)(ks)
            vs = Lambda(reshape1)(vs)

            if mask is not None:
                mask = Lambda(lambda x: k_back.repeat_elements(x, n_head, 0))(mask)
            head, attn = self.attention(qs, ks, vs, mask=mask)

            def reshape2(x):
                s = tf.shape(x)  # [n_head * batch_size, len_v, d_v]
                x = tf.reshape(x, [n_head, -1, s[1], s[2]])
                x = tf.transpose(x, [1, 2, 0, 3])
                x = tf.reshape(x, [-1, s[1], n_head * d_v])  # [batch_size, len_v, n_head * d_v]
                return x

            head = Lambda(reshape2)(head)
        elif self.mode == 1:
            heads = []
            attn_s = []
            for i in range(n_head):
                qs = self.qs_layers[i](q)
                ks = self.ks_layers[i](k)
                vs = self.vs_layers[i](v)
                head, attn = self.attention(qs, ks, vs, mask)
                heads.append(head)
                attn_s.append(attn)
            head = Concatenate()(heads) if n_head > 1 else heads[0]
            attn = Concatenate()(attn_s) if n_head > 1 else attn_s[0]

        outputs = self.w_o(head)
        outputs = Dropout(self.dropout)(outputs)
        return outputs, attn

    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({
            'n_head': self.n_head,
            'd_model': self.d_k * self.n_head,  # d_k = d_v
            'dropout': self.dropout,
            'mode': self.mode,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class PositionWiseFeedForward(Layer):
    def __init__(self, d_hid, d_inner_hid, dropout=0.1, **kwargs):
        super(PositionWiseFeedForward, self).__init__(**kwargs)
        self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
        self.w_2 = Conv1D(d_hid, 1)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)

    def call(self, x):
        output = self.w_1(x)
        output = self.w_2(output)
        output = self.dropout(output)
        output = Add()([output, x])
        return self.layer_norm(output)

    def get_config(self):
        config = super(PositionWiseFeedForward, self).get_config()
        config.update({
            'd_hid': self.w_2.filters,
            'd_inner_hid': self.w_1.filters,
            'dropout': self.dropout.rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class EncoderLayer(Layer):
    def __init__(self, d_model, d_inner_hid, n_head, dropout=0.1, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.self_att_layer = MultiHeadAttention(n_head, d_model, dropout=dropout, mode=1)
        self.pos_ffn_layer = PositionWiseFeedForward(d_model, d_inner_hid, dropout=dropout)
        self.norm_layer = LayerNormalization()

    def call(self, enc_input, mask=None):
        output, slf_attn = self.self_att_layer(enc_input, enc_input, enc_input, mask=mask)
        output = self.norm_layer(Add()([enc_input, output]))
        output = self.pos_ffn_layer(output)
        return output

    def get_config(self):
        config = super(EncoderLayer, self).get_config()
        config.update({
            'd_model': self.self_att_layer.d_k * self.self_att_layer.n_head,  # d_k = d_v
            'd_inner_hid': self.pos_ffn_layer.w_1.filters,
            'n_head': self.self_att_layer.n_head,
            'dropout': self.self_att_layer.dropout,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
