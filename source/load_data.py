import re
import os
import math
import numpy as np
import logging
from nltk import word_tokenize
from keras.preprocessing.sequence import pad_sequences

logger = logging.getLogger(__name__)
local_file = os.path.split(__file__)[-1]
logging.basicConfig(
    format='%(asctime)s : %(filename)s : %(funcName)s : %(levelname)s : %(message)s',
    level=logging.INFO)


def remove_punctuation(sentence):
    remove_chars = '[’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    result = re.sub(remove_chars, ' ', sentence)
    result = ' '.join(result.split())
    return result


def pad_sentence(sentence, max_len):
    seg_id = pad_sequences([sentence], maxlen=max_len, padding='post')
    return seg_id[0]


def tran2ids(words, vocabulary, max_len, use_pad=True):
    words = word_tokenize(words)
    word_id = [vocabulary.get(e, len(vocabulary) - 1) for e in words]
    if use_pad:
        word_id = pad_sentence(word_id, max_len)
    return word_id


def word_to_char_ids(word, char_vocab, max_len=25):
    char_ids = []
    for ch in word:
        char_ids.append(char_vocab.get(ch, len(char_vocab)-1))

    char_ids = pad_sentence(char_ids, max_len)
    return char_ids


def sentence_to_char_ids(setence, char_id_dict, sentence_len, word_len):
    # char id
    setence = word_tokenize(setence)
    char_ids = [word_to_char_ids(e, char_id_dict, max_len=word_len) for e in setence]
    # satisfy word max len
    if len(char_ids) > sentence_len:
        char_ids = char_ids[:sentence_len]
    else:
        for i in range(sentence_len - len(char_ids)):
            # if use [0] * char_max_len will make an error
            char_ids.append(np.zeros(word_len, dtype='int'))
    return char_ids


def repre_word(e_ids, embedding, default_embedding):
    default_embedding = np.array(default_embedding)
    repre = []
    for e_id in e_ids:
        if e_id in embedding:
            temp = np.array(embedding[e_id])
            repre.append(temp)
        else:
            repre.append(default_embedding)

    return np.array(repre)


def load_word_vocabulary(path, use_pad=True):
    vocab_dict = dict()
    word_list = []
    for line in open(path, encoding='utf8'):
        row = line.strip().split('\t')
        w_id = int(row[0])
        text = row[1]
        vocab_dict[text] = w_id
        word_list.append(text)
    vocab_dict['<pad>'] = 0
    vocab_dict['<sep>'] = len(vocab_dict)
    vocab_dict['<unk>'] = len(vocab_dict)
    word_list.append('<sep>')
    word_list.append('<unk>')
    if use_pad:
        word_list.insert(0, '<pad>')
    return vocab_dict, word_list


def load_number_mapping():
    number_dict = {}
    for line in open('../input/number_mapping.txt', encoding='utf8'):
        row = line.strip('\n').split('@')
        cardinal = row[0]
        number_dict[cardinal] = []
        number_dict[cardinal].append(cardinal)
        for diff_type_nums in row[1].split('|'):
            diff_type_nums = diff_type_nums.split('__')
            for num in diff_type_nums:
                num = str.lower(num)
                number_dict[cardinal].append(num)
    return number_dict


def load_char_vocabulary(path, use_pad=True):
    char_dict = dict()
    char_list = []
    for line in open(path, encoding='utf8'):
        row = line.strip('\n')
        row = row.split('\t')
        w_id = int(row[0])
        text = row[1]
        char_dict[text] = w_id
        char_list.append(text)
    char_dict['<pad>'] = 0
    char_dict['<sep>'] = len(char_dict)
    char_dict['<unk>'] = len(char_dict)
    char_list.append('<sep>')
    char_list.append('<unk>')
    if use_pad:
        char_list.insert(0, '<pad>')
    return char_dict, char_list


def load_train_data(path):
    data_dict = {}
    all_data = []
    with open(path, encoding='utf8')as f:
        lines = f.readlines()
        for line in lines:
            row = line.strip().split('\t')
            mention = row[3]
            label = row[2]
            data_dict[mention] = label
            all_data.append((row[0], mention, label))
    return data_dict, all_data


def load_data_by_id(datapath, word_dict, char_dict, max_len, char_max_len=25, mode='test'):
    all_data, raw_data = [], []
    with open(datapath, encoding='utf8')as f:
        lines = f.readlines()
        for line in lines:
            row = line.strip().split('\t')
            mention = word_tokenize(row[3])
            mention_id = [word_dict.get(w, len(word_dict)-1) for w in mention]
            mention_id = pad_sentence(mention_id, max_len)
            char_ids = sentence_to_char_ids(row[3], char_dict, max_len, char_max_len)
            label = row[2]
            if mode == 'train' and label == 'unmapped': continue
            if mode == 'train' and str.lower(label) == 'cui-less': continue
            all_data.append((mention_id, label, char_ids))
            raw_data.append((row[0], row[3], row[2]))
    logger.info('loaded {a}, the data size is {b}'.format(a=datapath, b=len(all_data)))
    return all_data, raw_data


def load_entity_by_id(path, word_dict, char_dict, max_len, char_max_len=25, use_pad=True):
    all_entity, entity_dict, raw_entity_dict = set(), {}, {}
    with open(path, encoding='utf8')as f:
        lines = f.readlines()
        for line in lines:
            row = line.strip().split('\t')
            row[1] = ' '.join(row[1].split())
            entity = word_tokenize(row[1])
            entity_id = [word_dict.get(e, len(word_dict)-1) for e in entity]
            if use_pad:
                entity_id = pad_sentence(entity_id, max_len)
            # char id
            char_ids = sentence_to_char_ids(row[1], char_dict, max_len, char_max_len)

            label = row[0]
            all_entity.add(label)
            if label not in entity_dict:
                entity_dict[label] = []
            entity_dict[label].append((entity_id, char_ids))

            if label not in raw_entity_dict:
                raw_entity_dict[label] = []
            raw_entity_dict[label].append((entity_id, row[1]))

    logger.info('loaded {a}, the data size is {b}'.format(a=path, b=len(list(all_entity))))
    return all_entity, entity_dict, raw_entity_dict


def load_entity(entity_path):
    entity_dict = {}
    id_map = {}
    with open(entity_path, encoding='utf8')as f:
        lines = f.readlines()
        for line in lines:
            row = line.strip().split('\t')
            id, entity = row[0], row[1]
            if id not in entity_dict:
                entity_dict[id] = []
            entity_dict[id].append(entity)
            if len(row) < 3:continue
            alt_ids = row[2].split('|')
            for alt_id in alt_ids:
                id_map[alt_id] = id
    return entity_dict, id_map


def load_candidates_by_id(word_dict, char_dict, max_len, can_path, max_word_len, topk=10, alpha=0.5):
    logging.info('load candidates from path = {a}, topk = {b}, alpha = {c}'.format(a=can_path, b=topk, c=alpha))
    can_dict, raw_can_dict, can_char_dict = {}, {}, {}
    with open(can_path, encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            row = line.strip().split('\t')
            mention = row[0]
            can_dict[mention] = []
            raw_can_dict[mention] = {}
            for index, cans in enumerate(row[1:topk+1]):
                cans = cans.split('__')
                score = 1.0
                if len(cans) == 3:
                    label, can_name, score = cans[0], cans[1], float(cans[2])
                if len(cans) == 2:
                    label, can_name = cans[0], cans[1]
                if score < alpha:continue
                raw_can_dict[mention][label] = can_name
                can = word_tokenize(can_name)
                can_id = [word_dict.get(e, len(word_dict) - 1) for e in can]
                can_id = pad_sentence(can_id, max_len)
                can_dict[mention].append((label, can_id))

                #char
                if can_name not in can_char_dict:
                    chars = sentence_to_char_ids(can_name, char_dict, max_len, max_word_len)
                    can_char_dict[can_name] = chars

    return can_dict, raw_can_dict, can_char_dict


def load_mention_of_each_entity(train_data_path):
    _, all_data = load_train_data(train_data_path)

    mention_of_entity = dict()
    for doc_id, mention, e_id in all_data:
        if e_id not in mention_of_entity:
            mention_of_entity[e_id] = []
        if mention not in mention_of_entity[e_id]:
            mention_of_entity[e_id].append(mention)
    return mention_of_entity


def load_mention_entity_prior2(path='../output/mention_entity_prior.txt'):

    mention_entity_dict = dict()
    with open(path, encoding='utf8')as f:
        lines = f.readlines()
        for line in lines:
            row = line.split('\t')
            mention, entity, cnt = row[0], row[1], int(row[2])
            if mention not in mention_entity_dict:
                mention_entity_dict[mention] = dict()
            mention_entity_dict[mention][entity] = math.log(cnt)
    return mention_entity_dict


def load_entity_emb(path):
    entity_emb_dict = dict()
    all_emb = list()
    with open(path)as f:
        lines = f.readlines()
        for line in lines:
            row = line.strip().split(' ')
            e_id = row[0]
            emb = row[1:]
            emb = [float(e) for e in emb]
            entity_emb_dict[e_id] = emb
            all_emb.append(emb)

    all_emb = np.array(all_emb)
    mean_emb = np.mean(all_emb, axis=0)
    logging.info('loaded embedding, size = {a}'.format(a=np.shape(all_emb)))
    return entity_emb_dict, mean_emb


def load_voting_eid_by_doc(raw_data, can_path, k=10):
    doc_mention_dict, mention_one_entity = dict(), dict()
    with open(can_path, encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            row = line.strip().split('\t')
            mention = row[0]
            can = row[1].strip().split('__')
            if len(can) == 3:
                c_id, c_name, c_score = can[0], can[1], float(can[2])
            else:
                c_id, c_name = can[0], can[1]
            mention_one_entity[mention] = c_id
    for doc_id, mention, label in raw_data:
        if doc_id not in doc_mention_dict:
            doc_mention_dict[doc_id] = list()
        if mention not in doc_mention_dict[doc_id]:
            doc_mention_dict[doc_id].append(mention)

    context_mention = dict()
    for doc_id, mention_list in doc_mention_dict.items():
        if doc_id not in context_mention:
            context_mention[doc_id] = dict()
        for index, mention in enumerate(mention_list):
            left = index-k if index-k >= 0 else 0
            left_k_mentions = mention_list[left:index]
            right = index+k if index+k < len(mention_list) else len(mention_list)-1
            right_k_mentions = mention_list[index+1:right+1]

            merge_mentions = left_k_mentions + right_k_mentions
            if left == 0 and len(merge_mentions) != 2*k:
                merge_mentions += mention_list[len(merge_mentions)-k*2:]
            if right == len(mention_list)-1 and len(merge_mentions) != 2*k:
                merge_mentions += mention_list[:k*2 - len(merge_mentions)]
            if len(merge_mentions) < 2*k:
                for i in range(2*k-len(merge_mentions)):
                    merge_mentions.append(merge_mentions[0])
            mention_first_can = [mention_one_entity[men] for men in merge_mentions]
            context_mention[doc_id][mention] = mention_first_can

    return context_mention


def get_embedding_of_voting(voting_eids, entity_embedding, default_embedding):
    voting_emb_dict = dict()
    for doc_id, mentions in voting_eids.items():
        if doc_id not in voting_emb_dict:
            voting_emb_dict[doc_id] = dict()
        for mention, votings in mentions.items():
            voting_emb = repre_word(votings, entity_embedding, default_embedding)
            voting_emb_dict[doc_id][mention] = voting_emb
    return voting_emb_dict


def load_mention_context(path, max_len, word_dict):
    mention_context_dict = dict()
    with open(path, encoding='utf8')as f:
        lines = f.readlines()
        for line in lines:
            row = line.strip().split('\t')
            doc_id, mention, context = row[0], row[1], row[2]
            context = word_tokenize(context)
            context = [word for word in context]
            entity_id = [word_dict.get(w, len(word_dict) - 1) for w in context]
            entity_id = pad_sentence(entity_id, max_len)

            if doc_id not in mention_context_dict:
                mention_context_dict[doc_id] = dict()
            mention_context_dict[doc_id][mention] = entity_id
    logging.info('loaded from {a}, data size = {b}'.format(a=path, b=len(mention_context_dict)))
    return mention_context_dict


if __name__ == '__main__':
    load_number_mapping()