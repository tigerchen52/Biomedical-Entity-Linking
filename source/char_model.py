import keras.backend as K
from keras.layers.core import Lambda
from keras.models import Sequential
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM


class CharRNN():
    def __init__(self, input_shape, dropout, rnn_dim, return_sequences=False):
        self.input_shape = input_shape
        self.rnn_dim = rnn_dim
        self.model = Sequential(name='char_rnn')
        self.model.add(
            Bidirectional(LSTM(self.rnn_dim,
                              dropout=dropout,
                              return_sequences=return_sequences),
                          input_shape=(self.input_shape[-2], self.input_shape[-1])))

    def __call__(self, char_embed):
        def backend_reshape1(x):
            return K.reshape(x, (-1, self.input_shape[-2], self.input_shape[-1]))

        char_embed = Lambda(backend_reshape1, output_shape=(self.input_shape[-2], self.input_shape[-1]))(char_embed)
        result = self.model(char_embed)

        def backend_reshape2(x):
            return K.reshape(x, (-1, self.input_shape[1], self.rnn_dim*2))
        result = Lambda(backend_reshape2, output_shape=(self.input_shape[1], self.rnn_dim*2))(result)
        return result

