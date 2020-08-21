import keras.backend as K
from keras.models import Sequential
from keras import regularizers, initializers, constraints, activations
import tensorflow as tf
from keras.layers import *
from keras.activations import softmax
from keras.engine.topology import Layer


class WordRepresLayer(object):
    def __init__(self, words_num, word_embedding_dim, sequence_length, embedding_matrix, mask_zero=False):

        self.model = Sequential()
        self.model.add(Embedding(words_num,
                                 word_embedding_dim,
                                 weights=[embedding_matrix],
                                 input_length=sequence_length,
                                 trainable=True,
                                 mask_zero=mask_zero, name='word_embedding'))

    def __call__(self, inputs):
        return self.model(inputs)


class CNNContextLayer(object):
    def __init__(self, filters, kernel_size, input_shape, dropout=0.5, init='glorot_uniform', padding='valid', use_max_pooling=True):
        self.use_max_pooling = use_max_pooling
        self.model = Sequential()
        conv = Conv1D(filters=filters, kernel_size=kernel_size,
                      input_shape=input_shape, padding=padding,
                      kernel_initializer=init, kernel_regularizer=regularizers.l2(1e-4), name='cnn_{a}'.format(a=kernel_size))
        self.model.add(conv)
        self.model.add(Activation('relu'))
        self.model.add(Dropout(dropout))
        if self.use_max_pooling:
            self.model.add(GlobalMaxPooling1D())

    def __call__(self, inputs):
        if not self.use_max_pooling:
            convolution = self.model(inputs)
            max_pooling = GlobalMaxPooling1D()(convolution)
            ave_pooling = GlobalAveragePooling1D()(convolution)
            return concatenate([max_pooling, ave_pooling])

        return self.model(inputs)


class Dot_Abs():
    '''
    dot layer and absolute layer
    '''
    def __init__(self, input_shape, return_con=False, axis=1):
        self.input_shape = input_shape
        self.axis = axis
        self.return_con = return_con
        self.model = Sequential()
        self.model.add(
            TimeDistributed(Dense(64, activation='relu')))
        self.model.add(TimeDistributed(Dropout(0.2)))

    def __call__(self, x1, x2):

        def _mult_ops(args):
            x1 = args[0]
            x2 = args[1]
            return x1 * x2

        def _sub_ops(args):
            x1 = args[0]
            x2 = args[1]
            x = K.abs(x1 - x2)
            return x

        output_shape = None
        if self.axis == 1:
            output_shape = (self.input_shape[1], )
        if self.axis == 2:
            output_shape = (self.input_shape[1], self.input_shape[2],)
        # (batch_size, timesteps, dim)
        mat = Lambda(
            _mult_ops,
            output_shape=output_shape)([x1, x2])

        sub = Lambda(_sub_ops, output_shape=output_shape)([x1, x2])
        result = concatenate([mat, sub])
        if self.return_con:
            result = mat, sub
        return result


class CosineLayer():
    '''
    calculate cosine similarity on the last dimension
    '''

    def __call__(self, x1, x2, axes=2):
        def _cosine(x):
            dot1 = K.batch_dot(x[0], x[1], axes=axes)
            dot2 = K.batch_dot(x[0], x[0], axes=axes)
            dot3 = K.batch_dot(x[1], x[1], axes=axes)
            max_ = K.maximum(K.sqrt(dot2 * dot3), K.epsilon())
            return dot1 / max_

        shape = K.int_shape(x1)
        output_shape = None
        if axes == 2:
            output_shape = (shape[1], shape[1],)
        if axes == 1:
            output_shape = (1,)
        # (batch_size, timesteps1, dim)
        res = Lambda(
            _cosine,
            output_shape=output_shape)([x1, x2])
        return res


class KMaxPooling(Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """

    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k

    def compute_output_shape(self, input_shape):
        return input_shape[0], (input_shape[2] * self.k)

    def call(self, inputs):
        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_input = tf.transpose(inputs, [0, 2, 1])

        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(shifted_input, k=self.k, sorted=True, name=None)[0]

        # return flattened output
        return Flatten()(top_k)


class ESIMLayer(Layer):

    def __init__(self, **kwargs):
        super(ESIMLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)

        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[0][2], input_shape[0][2]),
                                 initializer='uniform',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias',
                                 shape=(input_shape[0][2],),
                                 initializer='uniform',
                                 trainable=True)
        super(ESIMLayer, self).build(input_shape)

    def call(self, inputs, transpose=False):

        x, y = inputs[0], inputs[1]

        def _dot_product(args):
            x, y = args[0], args[1]
            x = K.dot(x, self.W) + self.b
            return K.batch_dot(x, K.permute_dimensions(y, (0, 2, 1)))

        def _normalize(args, transpose=False):
            att_w = args[0]
            x = args[1]
            if transpose:
                att_w = K.permute_dimensions(att_w, (0, 2, 1))
            e = K.exp(att_w - K.max(att_w, axis=-1, keepdims=True))
            sum_e = K.sum(e, axis=-1, keepdims=True)
            nor_e = e / sum_e
            return K.batch_dot(nor_e, x)

        att_w = _dot_product([x, y])
        outputs = _normalize([att_w, y], transpose)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


class ESIMAttention():

    def __init__(self, input_shape):
        self.sequence_length = input_shape[1]
        self.input_dim = input_shape[2]

    def __call__(self, x, y):

        def _normalize(args):
            arg_x, arg_y = args[0], args[1]
            e = K.batch_dot(arg_x, K.permute_dimensions(arg_y, (0, 2, 1)))
            sum_e = K.maximum(K.sum(e, axis=-1, keepdims=True), K.epsilon())
            nor_e = e / sum_e
            return K.batch_dot(nor_e, arg_y)


        output_shape = (self.sequence_length, self.input_dim,)
        #output_shape = (self.sequence_length, self.sequence_length,)
        # (batch_size, timesteps1, dim)
        att1 = Lambda(
            _normalize,
            output_shape=output_shape)([x, y])
        # (batch_size, timestep2, dim)
        att2 = Lambda(
            _normalize,
            output_shape=output_shape)([y, x])

        return att1, att2


class NormAttention():

    def __init__(self, input_shape):
        self.sequence_length = input_shape[1]
        self.input_dim = input_shape[2]

    def __call__(self, x, y):

        def _normalize(args):
            arg_x, arg_y = args[0], args[1]
            e = K.batch_dot(arg_x, arg_y, axes=-1)
            sum_e = K.maximum(K.sum(e, axis=-1, keepdims=True), K.epsilon())
            nor_e = e / sum_e
            return nor_e


        output_shape = (self.sequence_length,)

        weight = Lambda(
            _normalize,
            output_shape=output_shape)([x, y])

        return weight


class DenseContextLayer():

    def __init__(self, input_shape, dense_dim, dropout):
        self.model = Sequential()
        rnn = Bidirectional(GRU(dense_dim, return_sequences=True, input_shape=(input_shape[-2], input_shape[-1])))
        self.model.add(rnn)
        self.model.add(TimeDistributed(Activation('relu')))
        self.model.add(TimeDistributed(Dropout(dropout)))

    def __call__(self, x):
        return self.model(x)


class PoolingLayer(object):

    def __call__(self, inputs):
        max_pooling = GlobalMaxPooling1D()(inputs)
        ave_pooling = GlobalAveragePooling1D()(inputs)
        pooling = concatenate([max_pooling, ave_pooling])
        return pooling


def create_pretrained_embedding(pretrained_weights, trainable=True):
    "Create embedding layer from a pretrained weights array"
    in_dim, out_dim = pretrained_weights.shape
    embedding = Embedding(in_dim, out_dim, weights=[pretrained_weights], trainable=trainable, name='word_embedding')
    return embedding


def create_char_embedding(charsize, maxlen, max_char_len, char_embedding_dim):
    char_embedding = Embedding(input_dim=charsize, output_dim=char_embedding_dim,
                               embeddings_initializer='lecun_uniform', input_shape=(maxlen, max_char_len),
                               mask_zero=False, trainable=True, name='char_embedding')
    return char_embedding


def soft_attention_alignment(input_1, input_2):
    "Align text representation with neural soft attention"
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1))(attention)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2))(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned


class HingeLoss():
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def __call__(self, pos_score, neg_score):
        def f(args):
            basic_loss = self.alpha - args[0] + args[1]
            result = K.maximum(basic_loss, 0)
            return K.sum(result)

        out_shape = (1,)
        result = Lambda(f, output_shape=out_shape)([pos_score, neg_score])
        return result



