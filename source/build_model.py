import keras.backend as K
import numpy as np
import logging
import os
from keras.models import Model
from keras import optimizers
from layers import (CNNContextLayer, Dot_Abs, ESIMAttention, NormAttention, HingeLoss)
from char_model import CharRNN
from keras.layers import *

logger = logging.getLogger(__name__)
local_file = os.path.split(__file__)[-1]
logging.basicConfig(
    format='%(asctime)s : %(filename)s : %(funcName)s : %(levelname)s : %(message)s',
    level=logging.INFO)


def input_attention_hinge_loss(word_embedding_matrix, word_list, char_list, params):
    logger.info(params)
    words_num = len(word_list)
    char_num = len(char_list)
    word_embedding_dim = np.shape(word_embedding_matrix)[-1]
    sentence_length = params.sentence_length
    filters = params.filters
    kernel_sizes = [int(x) for x in params.kernel_sizes.split(',')]
    char_rnn_dim = params.char_rnn_dim
    dropout = params.dropout
    character_length = params.character_length
    char_dim = params.char_dim
    lr = params.lr
    hinge = params.hinge
    coherence_k = params.voting_k * 2
    context_sentence_length = params.context_sentence_length
    context_rnn_dim = params.context_rnn_dim

    # Build input
    mention = Input(batch_shape=(None, sentence_length), dtype='int32', name='mention')
    candidate = Input(batch_shape=(None, sentence_length), dtype='int32', name='candidate')
    men_char = Input(batch_shape=(None, sentence_length, character_length), dtype='int32', name='men_char')
    can_char = Input(batch_shape=(None, sentence_length, character_length), dtype='int32', name='can_char')
    can_prior = Input(batch_shape=(None, 1), dtype='float32', name='can_prior')
    can_context = Input(batch_shape=(None, context_sentence_length), dtype='int32', name='can_context')
    candidate_emb = Input(batch_shape=(None, 50), dtype='float32', name='candidate_emb')
    voting_candidates_emb = Input(batch_shape=(None, coherence_k, 50), dtype='float32', name='voting_candidates_emb')

    pos_candidate = Input(batch_shape=(None, sentence_length), dtype='int32', name='pos_candidate')
    pos_can_char = Input(batch_shape=(None, sentence_length, character_length), dtype='int32', name='pos_can_char')
    neg_candidate = Input(batch_shape=(None, sentence_length), dtype='int32', name='neg_candidate')
    neg_can_char = Input(batch_shape=(None, sentence_length, character_length), dtype='int32', name='neg_can_char')
    pos_can_prior = Input(batch_shape=(None, 1), dtype='float32', name='pos_can_prior')
    neg_can_prior = Input(batch_shape=(None, 1), dtype='float32', name='neg_can_prior')
    pos_candidate_emb = Input(batch_shape=(None, 50), dtype='float32', name='pos_candidate_emb')
    neg_candidate_emb = Input(batch_shape=(None, 50), dtype='float32', name='neg_candidate_emb')

    word_layer = Embedding(
        input_dim=words_num,
        output_dim=word_embedding_dim,
        weights=[word_embedding_matrix],
        trainable=True
    )
    w_res1 = word_layer(mention)
    w_res2 = word_layer(candidate)

    # char embedding
    char_layer = Embedding(char_num, char_dim, input_length=(sentence_length, character_length),
                           trainable=True, mask_zero=True,
                           name='char_embedding', embeddings_initializer='he_uniform')

    char_res1 = char_layer(men_char)
    char_res2 = char_layer(can_char)
    char_shape = K.int_shape(char_res1)

    #char_repre
    char_rnn = CharRNN(char_shape, dropout, char_rnn_dim)

    # batch * sequenth length * rnn dim
    ch_res1 = char_rnn(char_res1)
    ch_res2 = char_rnn(char_res2)

    w_res1 = concatenate([w_res1, ch_res1])
    w_res2 = concatenate([w_res2, ch_res2])

    #attention layer
    shape = K.int_shape(w_res1)
    attention1, attention2 = ESIMAttention(shape)(w_res1, w_res2)

    shape = K.int_shape(attention1)
    mention_att1 = Dot_Abs(shape, axis=2)(w_res1, attention1)
    repre1 = concatenate([w_res1, attention1], axis=-1)
    com_repre1 = concatenate([repre1, mention_att1], axis=-1)

    mention_att2 = Dot_Abs(shape, axis=2)(w_res2, attention2)
    repre2 = concatenate([w_res2, attention2], axis=-1)
    com_repre2 = concatenate([repre2, mention_att2], axis=-1)

    # Cnn context layer
    w_res1_shape = K.int_shape(com_repre1)
    cnns = [CNNContextLayer(filters=filters, kernel_size=kernel_size, input_shape=(w_res1_shape[1], w_res1_shape[2]),
                            dropout=dropout) for kernel_size in kernel_sizes]

    sequence1 = concatenate([cnn(com_repre1) for cnn in cnns])
    sequence2 = concatenate([cnn(com_repre2) for cnn in cnns])
    sequence = concatenate([sequence1, sequence2])


    #context
    context_rnn1 = Bidirectional(GRU(context_rnn_dim))
    can_repre = word_layer(candidate)
    can_repre = context_rnn1(can_repre)
    context_rnn2 = Bidirectional(GRU(context_rnn_dim, return_sequences=True))
    context_repre = word_layer(can_context)
    context_repre = context_rnn2(context_repre)
    norm_att = NormAttention(K.int_shape(context_repre))
    weight = norm_att(can_repre, context_repre)
    context_repre = Dot(axes=1)([weight, context_repre])
    context_score = Dot(axes=1, normalize=True)([can_repre, context_repre])


    #cohrence
    candidate_embs = RepeatVector(K.int_shape(voting_candidates_emb)[1])(candidate_emb)
    coherence_feature = Dot(normalize=True, axes=-1)([candidate_embs, voting_candidates_emb])
    pool = GlobalAveragePooling1D()(coherence_feature)
    #undo
    coherence_feature = Dense(1)(pool)

    sequence = concatenate([sequence, context_score, coherence_feature, can_prior])

    #output
    sequence = Dropout(dropout)(sequence)
    dense = Dense(64, activation='relu', input_shape=K.int_shape(sequence))(sequence)
    dense = Dropout(dropout)(dense)
    pred = Dense(1, activation='sigmoid', name='final_dense')(dense)

    #hinge loss
    score_model = Model(
        inputs=[mention, candidate, men_char, can_char, can_prior, candidate_emb, voting_candidates_emb, can_context], outputs=pred)

    pos_score = score_model(
        [mention, pos_candidate, men_char, pos_can_char, pos_can_prior, pos_candidate_emb, voting_candidates_emb, can_context])
    neg_score = score_model(
        [mention, neg_candidate, men_char, neg_can_char, neg_can_prior, neg_candidate_emb, voting_candidates_emb, can_context])

    result = HingeLoss(alpha=hinge)(pos_score, neg_score)

    train_inputs = (
    mention, pos_candidate, men_char, pos_can_char, neg_candidate, neg_can_char, pos_can_prior, neg_can_prior,
    pos_candidate_emb, neg_candidate_emb, voting_candidates_emb, can_context)
    train_model = Model(inputs=train_inputs, outputs=result)

    # Compile model
    nadam = optimizers.nadam(lr=lr)
    train_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=nadam)
    pre_inputs = (mention, candidate, men_char, can_char, can_prior, candidate_emb, voting_candidates_emb, can_context)
    predict_model = Model(inputs=pre_inputs, outputs=pred)
    predict_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=nadam)

    return train_model, predict_model


if __name__ == '__main__':
    pass