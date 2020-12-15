import matplotlib.pyplot as plt
import random
import os
import numpy as np
import tensorflow as tf
import logging
from load_data import load_train_data, load_data_by_id, load_entity_by_id, \
            load_candidates_by_id, load_mention_entity_prior2, load_entity_emb, load_voting_eid_by_doc, \
            get_embedding_of_voting, load_mention_context, load_word_vocabulary, load_char_vocabulary
import matplotlib
matplotlib.use('agg')


logger = logging.getLogger(__name__)
local_file = os.path.split(__file__)[-1]
logging.basicConfig(
    format='%(asctime)s : %(filename)s : %(funcName)s : %(levelname)s : %(message)s',
    level=logging.INFO)


def set_random_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)


def sample(sample_list):
    return random.sample(sample_list, 1)[0]


def negative_sample(pos_id_list, id_list, sample_num):
    neg_reault = []
    cnt = 0
    while cnt < sample_num:
        negative_id = ''
        while negative_id == '' or negative_id in pos_id_list:
            negative_id = sample(id_list)
        neg_reault.append(negative_id)
        cnt += 1
    return neg_reault


def gen_valid_data(data_path, word_dict, char_dict, sentence_length, can_path, prior_path, entity_embedding_path, character_length,
                   topk=25, alpha=0.0, voting_k=10, context_path=None, context_max_len=100):
    all_data, raw_data = load_data_by_id(data_path, word_dict, char_dict, sentence_length, character_length)
    candidate_dict, raw_can_dict, can_char_dict = load_candidates_by_id(word_dict, char_dict, sentence_length, can_path, character_length, topk, alpha=alpha)
    mention_entity_prior = load_mention_entity_prior2(prior_path)

    #coherence
    entity_embedding, default_embedding = load_entity_emb(entity_embedding_path)
    voting_eids = load_voting_eid_by_doc(raw_data, can_path, voting_k)
    voting_emb_dict = get_embedding_of_voting(voting_eids, entity_embedding, default_embedding)

    # context
    mention_context_dict = load_mention_context(context_path, context_max_len, word_dict)

    mention_list, candidate_list, y_list, raw_can_name_list, x_char_list, y_char_list, can_prior = [], [], [], [], [], [], []
    can_emb_list, voting_emb_list = [], []
    context_list = []

    miss_cnt = 0
    test_data = list()
    for index, (mention_id, labels, char_ids) in enumerate(all_data):

        label = labels
        raw = raw_data[index]
        doc_id, raw_mention, raw_label = raw[0], raw[1], raw[2]

        mention_name = raw_mention
        can_list_of_the_mention = candidate_dict[mention_name]
        raw_can_dict_of_the_mention = raw_can_dict[mention_name]

        for can_label, can_id in can_list_of_the_mention:
            mention_list.append(mention_id)
            candidate_list.append(can_id)
            y_list.append(can_label)

            entity_name = raw_can_dict_of_the_mention[can_label]
            raw_can_name_list.append(entity_name)

            #char
            x_char_list.append(char_ids)
            can_char_ids = can_char_dict[entity_name]
            y_char_list.append(can_char_ids)

            #prior
            prior_value = 0
            if mention_name in mention_entity_prior and can_label in mention_entity_prior[mention_name]:
                prior_value = mention_entity_prior[mention_name][can_label]
            can_prior.append([prior_value])

            #coherence
            can_emb = default_embedding
            if can_label in entity_embedding:
                can_emb = entity_embedding[can_label]
            voting_emb = voting_emb_dict[doc_id][raw_mention]
            can_emb_list.append(can_emb)
            voting_emb_list.append(voting_emb)

            # context
            doc_id = raw_data[index][0]
            mention_context = [0] * context_max_len
            if doc_id in mention_context_dict and mention_name in mention_context_dict[doc_id]:
                mention_context = mention_context_dict[doc_id][mention_name]
            context_list.append(mention_context)

        data = ({'mention': np.array(mention_list), 'candidate': np.array(candidate_list),
               'entity_name': raw_can_name_list, 'men_char': np.array(x_char_list),
               'can_char': np.array(y_char_list), 'can_prior':np.array(can_prior),
                 'candidate_emb':np.array(can_emb_list), 'voting_candidates_emb':np.array(voting_emb_list), 'can_context':np.array(context_list)
               }, np.array(y_list), (label, doc_id, raw_mention))

        test_data.append(data)
        mention_list, candidate_list, y_list, raw_can_name_list, x_char_list, y_char_list, can_prior = [], [], [], [], [], [], []
        can_emb_list, voting_emb_list = [], []
        context_list = []

    logging.info('test data size = {a}, miss data size = {b}'.format(a=len(all_data), b=miss_cnt))
    return test_data


def gen_train_data(data_path, word_dict, char_dict, entity_path, batch_size, sentence_length, character_length, can_path, prior_path, entity_embedding_path,
                         topk=10, alpha=0.0, voting_k=10, context_path=None, context_max_len=100):

    all_data, raw_data = load_data_by_id(data_path, word_dict, char_dict, sentence_length, character_length, mode='train')
    all_entity, entity_dict, _ = load_entity_by_id(entity_path, word_dict, char_dict, sentence_length, character_length)
    candidate_dict, raw_can_dict, can_char_dict = load_candidates_by_id(word_dict, char_dict, sentence_length, can_path, character_length, topk=topk, alpha=alpha)
    #cohrence
    entity_embedding, default_embedding = load_entity_emb(entity_embedding_path)
    voting_eids = load_voting_eid_by_doc(raw_data, can_path, voting_k)
    voting_emb_dict = get_embedding_of_voting(voting_eids, entity_embedding, default_embedding)

    #context
    mention_context_dict = load_mention_context(context_path, context_max_len, word_dict)

    mention_entity_prior = load_mention_entity_prior2(prior_path)
    mention_list, candidate_list, neg_candidate_list, x_char_list, y_char_list, z_char_list, pos_candidate_prior, neg_candidate_prior \
        = [], [], [], [], [], [], [], []
    pos_can_emb_list, neg_can_emb_list, voting_emb_list = [], [], []
    context_list = []

    while True:
        for index, (mention_id, label, char_ids) in enumerate(all_data):

            if label not in entity_dict:
                pos_sample, pos_y_chars = mention_id, char_ids
            else:
                synonyms = entity_dict[label]
                pos_sample, pos_y_chars = sample(synonyms)

            # use candidates to train
            mention = raw_data[index][1]
            candidates_of_this_mention = candidate_dict[mention]
            can_lables = [e1 for (e1, e2) in candidates_of_this_mention]
            if len(can_lables) == 1:
                neg_lables = negative_sample(label, list(all_entity), 1)[0]
            else:
                neg_lables = negative_sample(label, can_lables, 1)[0]

            neg_synonyms = entity_dict[neg_lables]
            neg_sample, neg_y_chars = sample(neg_synonyms)

            mention_list.append(mention_id)
            candidate_list.append(pos_sample)
            x_char_list.append(char_ids)
            y_char_list.append(pos_y_chars)

            neg_candidate_list.append(neg_sample)
            z_char_list.append(neg_y_chars)

            pos_prior_value, neg_prior_value = 0, 0
            if mention in mention_entity_prior and label in mention_entity_prior[mention]:
                pos_prior_value = mention_entity_prior[mention][label]
            if mention in mention_entity_prior and neg_lables in mention_entity_prior[mention]:
                neg_prior_value = mention_entity_prior[mention][neg_lables]
            pos_candidate_prior.append([pos_prior_value])
            neg_candidate_prior.append([neg_prior_value])

            #coherence
            doc_id = raw_data[index][0]
            pos_can_emb, neg_can_emb = default_embedding, default_embedding
            if label in entity_embedding:
                pos_can_emb = entity_embedding[label]
            if neg_lables in entity_embedding:
                neg_can_emb = entity_embedding[neg_lables]
            voting_emb = voting_emb_dict[doc_id][mention]
            pos_can_emb_list.append(pos_can_emb)
            neg_can_emb_list.append(neg_can_emb)
            voting_emb_list.append(voting_emb)

            #context
            doc_id = raw_data[index][0]
            if mention in mention_context_dict[doc_id]:
                mention_context = mention_context_dict[doc_id][mention]
            else: mention_context = [0]*context_max_len
            context_list.append(mention_context)

            if len(mention_list) % batch_size == 0:
                yield {'mention': np.array(mention_list), 'pos_candidate':np.array(candidate_list),
                       'men_char':np.array(x_char_list), 'pos_can_char':np.array(y_char_list)
                       ,'neg_candidate':np.array(neg_candidate_list), 'neg_can_char':np.array(z_char_list),
                       'pos_can_prior':np.array(pos_candidate_prior), 'neg_can_prior':np.array(neg_candidate_prior)
                       ,'pos_candidate_emb':np.array(pos_can_emb_list), 'neg_candidate_emb':np.array(neg_can_emb_list),
                       'voting_candidates_emb':np.array(voting_emb_list), 'can_context':np.array(context_list)}, \
                      np.array(mention_list),
                mention_list, candidate_list, neg_candidate_list, x_char_list, y_char_list, z_char_list, pos_candidate_prior, neg_candidate_prior \
                    = [], [], [], [], [], [], [], []

                pos_can_emb_list, neg_can_emb_list, voting_emb_list = [], [], []
                context_list = []


def make_loss_picture(history):
    print('Plot validation accuracy and loss...')

    # acc=history.history['acc']
    # val_acc=history.history['val_acc']
    loss=history.history['loss']
    val_loss=history.history['val_loss']

    # plt.plot(acc, label='acc')
    # plt.plot(val_acc, label='val_acc')
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'valid'], loc='upper left')
    # plt.savefig('../checkpoints/acc.png')
    # plt.close()

    plt.plot(loss, label='loss')
    plt.plot(val_loss, label='val_loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig('../checkpoints/loss.png')


if __name__ == '__main__':
    can_path = '../output/adr/candidates/training_aligned_cos_with_mention_candidate.txt'
    context_path = '../output/adr/context/train_mention_context.txt'
    word_vocab_path = '../output/adr/word_vocabulary.dict'
    char_vocab_path = '../output/adr/char_vocabulary.dict'
    prior_path = '../output/adr/mention_entity_prior.txt'
    word_dict, word_list = load_word_vocabulary(word_vocab_path, True)
    char_dict, char_list = load_char_vocabulary(char_vocab_path)
    data = gen_train_data(data_path='../output/adr/train_data.txt', word_dict=word_dict, char_dict=char_dict,
                                entity_path='../output/adr/entity_kb.txt', batch_size=6,topk=20, alpha=0.0,
                                sentence_length=20, character_length=25, can_path=can_path,
                                prior_path=prior_path,
                                entity_embedding_path='../output/adr/embed/entity_emb_50.txt',
                                context_path=context_path)
    cnt = 0
    for train, y in data:
        cnt += 1
        mention, pos_candidate, neg_candidate = train['mention'], train['pos_candidate'], train['neg_candidate']
        men_char, pos_can_char, neg_can_char = train['men_char'], train['pos_can_char'], train['neg_can_char']
        voting = train['can_context']
        print(mention)
        if len(np.shape(mention)) != 2:
            print(mention)
            raise Exception('error')