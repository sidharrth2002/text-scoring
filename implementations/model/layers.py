import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers as initializers, regularizers, constraints

'''
There is a set maximum length for keywords, depending on the dataset used.
'''
def get_token_num_for_keywords(group):
    if group == 'set3':
        return 15
    elif group == 'set4':
        return 15
    elif group == 'set5':
        return 13
    elif group == 'set6':
        return 10
    elif group == 'practice-a':
        return 15
    elif group == 'practice-b':
        return 15

# pytorch layer
'''
Nagappan et al. Pytorch implementation based on the Tensorflow implementation of Wang et al. (2019)
'''
class KeyAttention(nn.Module):
    """
    Compute attention between two sentences (S1(w1, e), S2(w2, e)) on word
    level (W(w1, w2), where w1 and w2 are the number of words in each sentence)
    Return: (att(scarlar, ), att_softmax(word_num, ))

    # Arguments
        op: The way to compute the word-level-attention.
            dp: Dot product. No weight for this approach.
                W = dot_product(S1, S2^T)
            sdp: Dot product with normalization (scaled dot product),
                [Vaswani, 2017]: W = dot_product(S1, S2^T)/sqrt(e)
            gen: General [Luong, 2015], W = dot_product(S1, M, S2^T),
                M is the weights to learn
            con: Concat [Bahdanau, 2015], W = dot_product(
                                                v,
                                                tanh(dot_product(M, [S1; S2]))
                                                )
                 where v and M are weights to learn.
        seed: random seed for initializing weights when it's needed.
              If seed = -1, then a identity matrix will be used
              for initialization.
        emb_dim: Dimension of word embeddings.
        word_att_pool: {max|sum|mean}, the pooling operation for
                       word-level attention.
        merge_ans_key: {concat|mean}
        beta: Bool.
    """
    def __init__(self,
                 name='key_attention',
                 op='dp',
                 seed=-1,
                 emb_dim=300,
                 word_att_pool='max',
                 merge_ans_key='concat',
                 beta=False,
                 batch_size=32,
                 tabular_config=None,
                 **kwargs):
        super(KeyAttention, self).__init__(**kwargs)
        self.op = op
        self.seed = seed
        self.emb_dim = emb_dim
        self.word_att_pool = word_att_pool
        self.merge_ans_key = merge_ans_key
        self.beta = beta
        self.W = None
        self.M = None
        self.v = None
        self.bias = None
        self.token_num_key = get_token_num_for_keywords(tabular_config['group'])
        self.token_num_ans = tabular_config.num_words
        self.mask_pad = True
        self.batch_size = batch_size

    def bdot(self, a, b):
        return torch.bmm(a, b)

    def softmax(self, x, mask):
        y = torch.exp(x - torch.max(x, axis=1, keepdim=True))
        sum_y = torch.bmm(y, torch.permute(mask, (0, 2, 1)))
        return y/sum_y

    def forward(self, inputs):
        # Attention matrix W(batch, w, w)

        ans, mask_ans, key, mask_key = inputs
        mask_ans_inf = torch.abs(mask_ans - 1) * -10000
        mask_key_inf = torch.abs(mask_key - 1) * -10000

        mask_ans_inf_1 = torch.unsqueeze(mask_ans_inf, 1)
        mask_key_inf_1 = torch.unsqueeze(mask_key_inf, 1)

        mask_ans_2 = torch.unsqueeze(mask_ans, 2)
        mask_key_2 = torch.unsqueeze(mask_key, 2)

        ans = ans * mask_ans_2
        key = key * mask_key_2

        Z_dp = torch.bmm(key, torch.permute(ans, (0, 2, 1)))

        norm_ans = torch.sqrt(torch.maximum(torch.sum(torch.square(ans), -1), torch.tensor(1e-7)))
        norm_key = torch.sqrt(torch.maximum(torch.sum(torch.square(key), -1), torch.tensor(1e-7)))

        norm_repeat_ans = torch.repeat_interleave(norm_ans, self.token_num_key, dim=0).reshape(self.batch_size, self.token_num_key, self.token_num_ans)
        norm_repeat_key = torch.repeat_interleave(norm_key, self.token_num_ans, dim=0).reshape(self.batch_size, self.token_num_ans, self.token_num_key)

        norm_repeat_key = torch.permute(norm_repeat_key, (0, 2, 1))

        Z_cos = Z_dp / (norm_repeat_key * norm_repeat_ans)

        if self.op == "dp":
            Z = Z_dp
        elif self.op == "sdp":
            Z = Z_dp / torch.sqrt(self.emb_dim)
        elif self.op == "gen":
            Z = torch.dot(key, self._M)
            Z = torch.bmm(Z, torch.permute(ans, (0, 2, 1)))
        elif self.op == "cos":
            Z = Z_cos

        Z_key = torch.permute(Z, (0, 2, 1))
        if self.mask_pad:
            Z_softmax_key = torch.softmax(Z_key + mask_key_inf_1, axis=2)
        else:
            Z_softmax_key = torch.softmax(Z_key, axis=2)

        V = torch.bmm(Z_softmax_key, key)
        V = V * mask_ans_2

        Z_ans = Z
        if self.mask_pad:
            Z_softmax_ans = torch.softmax(Z_ans + mask_ans_inf_1, axis=2)
        else:
            Z_softmax_ans = torch.softmax(Z_ans, axis=2)

        U = torch.bmm(Z_softmax_ans, ans)
        U = U * mask_key_2

        beta_key = torch.sigmoid(torch.max(Z_cos + mask_ans_inf_1, axis=2)[0] * 5)
        beta_key = torch.unsqueeze(beta_key, 2)

        Z_cos = torch.permute(Z_cos, (0, 2, 1))
        beta_ans = torch.sigmoid(torch.max(Z_cos + mask_key_inf_1, axis=2)[0] * 5)

        beta_ans = torch.unsqueeze(beta_ans, 2)

        if self.beta:
            U = U * beta_key
            V = V * beta_ans

        if self.word_att_pool == "sum":
            v = torch.sum(V, 1, keepdims=False)
            u = torch.sum(U, 1, keepdims=False)
        elif self.word_att_pool == "max":
            v = torch.max(V, 1, keepdims=False)
            u = torch.max(U, 1, keepdims=False)
        elif self.word_att_pool == "mean":
            v = torch.sum(V, 1, keepdims=False) / torch.sum(mask_ans_2, 1)
            u = torch.sum(U, 1, keepdims=False) / torch.sum(mask_key_2, 1)
        else:
            raise TypeError(
                "The pooling method need to be 'max', 'sum' or 'mean'!"
            )

        if self.merge_ans_key == 'concat':
            f = torch.cat([u, v], 1)
        elif self.merge_ans_key == 'mean':
            f = (u + v) / 2
        elif self.merge_ans_key == 'ans':
            f = u
        elif self.merge_ans_key == 'key':
            f = v

        Z_softmax_key = torch.permute(Z_softmax_key, (0, 2, 1))

        beta_ans = torch.unsqueeze(torch.squeeze(beta_ans, 2), 1)
        beta_key = torch.unsqueeze(torch.squeeze(beta_key, 2), 1)
        rtn_list = [f, Z, Z_softmax_ans, Z_softmax_key, beta_ans, beta_key]
        return rtn_list

# This is a replication of the Keras backend layer for Pytorch
class LambdaLayer(nn.Module):
    def __init__(self, lambd, name):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
        self.name = name
    def forward(self, x):
        return self.lambd(x)

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatibl|e with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

'''
Implementation from Riordan et al. (2021)
'''
class Attention(Layer):
    def __init__(self, op='attsum', activation='tanh', init_stdev=0.01, **kwargs):
        self.supports_masking = True
        assert op in {'attsum', 'attmean'}
        assert activation in {None, 'tanh'}
        self.op = op
        self.activation = activation
        self.init_stdev = init_stdev
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        init_val_v = (np.random.randn(input_shape[2]) * self.init_stdev).astype(K.floatx())
        self.att_v = K.variable(init_val_v, name='att_v')
        init_val_W = (np.random.randn(input_shape[2], input_shape[2]) * self.init_stdev).astype(K.floatx())
        self.att_W = K.variable(init_val_W, name='att_W')
        self.trainable_weights.append(self.att_v)
        self.trainable_weights.append(self.att_W)
        self.built = True

    def call(self, x, mask=None):
        y = K.dot(x, self.att_W)
        if not self.activation:
            weights = tf.tensordot(self.att_v, y, axes=[[0], [2]])
        elif self.activation == 'tanh':
            weights = tf.tensordot(self.att_v, K.tanh(y), axes=[[0], [2]])

        weights = K.softmax(weights)
        out = x * K.permute_dimensions(K.repeat(weights, x.shape[2]), [0, 2, 1])
        if self.op == 'attsum':
            out = K.sum(out, axis=1)
        elif self.op == 'attmean':
            out = out.sum(axis=1) / mask.sum(axis=1, keepdims=True)
        return K.cast(out, K.floatx())

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def compute_mask(self, x, mask):
        return None

    def get_config(self):
        config = {'op': self.op, 'activation': self.activation, 'init_stdev': self.init_stdev}
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

'''
ZeroMaskedEntries implementation based on Riordan et al. (2021)
'''
class ZeroMaskedEntries(Layer):
    """
    This layer is called after an Embedding layer.
    It zeros out all of the masked-out embeddings.
    It also swallows the mask without passing it on.
    You can change this to default pass-on behavior as follows:
    def compute_mask(self, x, mask=None):
        if not self.mask_zero:
            return None
        else:
            return K.not_equal(x, 0)
    """

    def __init__(self, **kwargs):
        self.support_mask = True
        super(ZeroMaskedEntries, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[1]
        self.repeat_dim = input_shape[2]

    def call(self, x, mask=None):
        mask = K.cast(mask, 'float32')
        mask = K.repeat(mask, self.repeat_dim)
        mask = K.permute_dimensions(mask, (0, 2, 1))
        return x * mask

    def compute_mask(self, input_shape, input_mask=None):
        return None

class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

class AttentionLayer(Layer):
    """
    Hierarchial Attention Layer as described by Hierarchical Attention Networks for Document Classification(2016)
    - Yang et. al.
    Source: https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf
    Theano backend
    """
    def __init__(self,attention_dim=100,return_coefficients=False,**kwargs):
        # Initializer
        self.supports_masking = True
        self.return_coefficients = return_coefficients
        self.init = initializers.get('glorot_uniform') # initializes values with uniform distribution
        self.attention_dim = attention_dim
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Builds all weights
        # W = Weight matrix, b = bias vector, u = context vector
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)),name='W')
        self.b = K.variable(self.init((self.attention_dim, )),name='b')
        self.u = K.variable(self.init((self.attention_dim, 1)),name='u')
        self._trainable_weights = [self.W, self.b, self.u]

        super(AttentionLayer, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, hit, mask=None):
        # Here, the actual calculation is done
        uit = K.bias_add(K.dot(hit, self.W),self.b)
        uit = K.tanh(uit)

        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)
        ait = K.exp(ait)

        if mask is not None:
            ait *= K.cast(mask, K.floatx())

        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = hit * ait

        if self.return_coefficients:
            return [K.sum(weighted_input, axis=1), ait]
        else:
            return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        if self.return_coefficients:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[-1], 1)]
        else:
            return input_shape[0], input_shape[-1]