import os
import sys
import argparse
import logging
import numpy as np
import load_data
from data_utils import gen_valid_data, gen_train_data
from build_model import input_attention_hinge_loss
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from predict import predict_data
from embed import generate_word_embeddings

#hyper-parameters
parser = argparse.ArgumentParser(description='biomedical entity liking')
parser.add_argument('-dataset', help='which benchmark to use, ncbi, clef, and adr ', type=str, default='ncbi')
parser.add_argument('-sentence_length', help='the length of mentions and entities', type=int, default=20)
parser.add_argument('-character_length', help='the length of characters', type=int, default=25)
parser.add_argument('-batch_size', help='the number of samples in one batch', type=int, default=64)
parser.add_argument('-develop_step', help='total number of steps to yield from validation_data generator', type=int, default=1500)
parser.add_argument('-filters', help='the dimension of CNN output', type=int, default=32)
parser.add_argument('-kernel_sizes', help='the length of the 1D convolution window', type=str, default='1,2,3')
parser.add_argument('-dropout', help='the rate of neurons are ignored during training', type=float, default=0.1)
parser.add_argument('-char_dim', help='the dimension of character embedding', type=int, default=128)
parser.add_argument('-lr', help='learning rate', type=float, default=5e-4)
parser.add_argument('-char_rnn_dim', help='the dimension of Bi-RNN for characters', type=int, default=64)
parser.add_argument('-hinge', help='the parameter for hinge loss', type=float, default=0.1)
parser.add_argument('-topk_candidates', help='the number of entity candidates', type=int, default=20)
parser.add_argument('-alpha', help='the threshold to filter candidates', type=float, default=0.0)
parser.add_argument('-voting_k', help='the number of adjacent mentions are used to calculate the coherence ', type=int, default=8)
parser.add_argument('-context_sentence_length', help='the length of context sentence', type=int, default=50)
parser.add_argument('-context_rnn_dim', help='the dimension of Bi-RNN for context words', type=int, default=32)
parser.add_argument('-epochs', help='the number of epochs to train the model', type=int, default=32)
parser.add_argument('-random_init', help='whether use initial word embeddings randomly', type=bool, default=True)
args = parser.parse_args()

#file path
origin_entity_path = '../input/{a}/origin_entity.txt'.format(a=args.dataset)
origin_train_path = '../input/{a}/origin_train.txt'.format(a=args.dataset)
origin_develop_path = '../input/{a}/origin_develop.txt'.format(a=args.dataset)
origin_test_path = '../input/{a}/origin_test.txt'.format(a=args.dataset)
train_data_path = '../output/{a}/train_data.txt'.format(a=args.dataset)
develop_data_path = '../output/{a}/develop_data.txt'.format(a=args.dataset)
test_data_path = '../output/{a}/test_data.txt'.format(a=args.dataset)
all_data_path = '../output/{a}/all_data.txt'.format(a=args.dataset)
word_vocab_path = '../output/{a}/word_vocabulary.dict'.format(a=args.dataset)
char_vocab_path = '../output/{a}/char_vocabulary.dict'.format(a=args.dataset)
bin_embedding_file = '../input/BioWordVec_PubMed_MIMICIII_d200.vec.bin'
txt_embedding_file = '../input/word_embedding.txt'
word_embedding_file = '../output/{a}/embed/word2vec_200_dim_with_context.npy'.format(a=args.dataset)
train_context_path = '../output/{a}/context/train_mention_context.txt'.format(a=args.dataset)
develop_context_path = '../output/{a}/context/develop_mention_context.txt'.format(a=args.dataset)
test_context_path = '../output/{a}/context/test_mention_context.txt'.format(a=args.dataset)
prior_path = '../output/{a}/mention_entity_prior.txt'.format(a=args.dataset)
entity_path = '../output/{a}/entity_kb.txt'.format(a=args.dataset)
entity_embedding_path = '../output/{a}/embed/entity_emb_50.txt'.format(a=args.dataset)
all_candidate_path = '../output/{a}/candidates/training_aligned_cos_with_mention_candidate.txt'.format(a=args.dataset)
test_can_path = '../output/{a}/candidates/test_candidates.txt'.format(a=args.dataset)
model_path = '../checkpoints/mp_lrs.h5'
log_path = '../checkpoints/model_log.txt'
predict_result_path = '../checkpoints/predict_result.txt'
predict_score_path = '../checkpoints/predict_score.txt'
model_weights_path = '../checkpoints/predict/predict_model_weights.h5'

logger = logging.getLogger(__name__)
local_file = os.path.split(__file__)[-1]
logging.basicConfig(
    format='%(asctime)s : %(filename)s : %(funcName)s : %(levelname)s : %(message)s',
    level=logging.INFO)


def create_model_and_fit(params):

    #load file
    word_dict, word_list = load_data.load_word_vocabulary(word_vocab_path, True)
    char_dict, char_list = load_data.load_char_vocabulary(char_vocab_path)

    if not params.random_init:
        word_embedding_matrix = np.load(word_embedding_file)
    else:
        word_embedding_matrix = np.random.rand(len(word_list), 200)

    #parameter
    batch_size = params.batch_size
    sentence_length = params.sentence_length
    character_length = params.character_length
    develop_step = params.develop_step
    topk_candidates = params.topk_candidates
    alpha = params.alpha
    voting_k = params.voting_k
    context_sentence_length = params.context_sentence_length


    #data
    train_data = gen_train_data(
        train_data_path,
        word_dict,
        char_dict,
        entity_path,
        batch_size,
        sentence_length,
        character_length,
        all_candidate_path,
        prior_path,
        entity_embedding_path,
        topk=topk_candidates,
        alpha=alpha,
        voting_k=voting_k,
        context_path=train_context_path,
        context_max_len=context_sentence_length
    )

    develop_data = gen_train_data(
        develop_data_path,
        word_dict,
        char_dict,
        entity_path,
        batch_size,
        sentence_length,
        character_length,
        all_candidate_path,
        prior_path,
        entity_embedding_path,
        topk=topk_candidates,
        alpha=alpha,
        voting_k=voting_k,
        context_path=develop_context_path,
        context_max_len=context_sentence_length
    )

    test_data = gen_valid_data(
        test_data_path,
        word_dict,
        char_dict,
        sentence_length,
        test_can_path,
        prior_path,
        entity_embedding_path,
        character_length,
        topk=topk_candidates,
        alpha=alpha,
        voting_k=voting_k,
        context_path=test_context_path,
        context_max_len=context_sentence_length
    )

    #create model
    train_model, predict_model = input_attention_hinge_loss(
        word_embedding_matrix,
        word_list,
        char_list,
        params=params
    )


    #fit model
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    logger.info(train_model.summary())
    history = train_model.fit_generator(
        generator=train_data,
        steps_per_epoch=500,
        epochs=30,
        verbose=1,
        validation_data=develop_data,
        validation_steps=develop_step,
        shuffle=True,
        callbacks=[
            ModelCheckpoint(model_path, save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True),
            CSVLogger(log_path),
            es
        ]
    )

    #predict
    predict_model.load_weights(model_weights_path)
    acc = predict_data(test_data, entity_path, predict_model, predict_result_path, predict_score_path)

    return acc


def run():
    if not os.path.exists(word_embedding_file):
        _, word_list = load_data.load_word_vocabulary(word_vocab_path, True)
        generate_word_embeddings(bin_embedding_file, txt_embedding_file, word_list, word_embedding_file)

    create_model_and_fit(args)


if __name__ == '__main__':
    logger.info("running %s", " ".join(sys.argv))
    run()