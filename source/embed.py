import random
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)
local_file = os.path.split(__file__)[-1]
logging.basicConfig(
    format='%(asctime)s : %(filename)s : %(funcName)s : %(levelname)s : %(message)s',
    level=logging.INFO)


def load_embeddings(path, emb_size, word_list):
    word_embed_dict = dict()
    with open(path)as f:
        line = f.readline()
        while line:
            row = line.strip().split(' ')
            if row[0] in word_list:
                word_embed_dict[row[0]] = row[1:emb_size+1]
            line = f.readline()
    logging.info('load word embedding, size = {a}'.format(a=len(word_embed_dict)))
    return word_embed_dict


def random_emb_value():
    return random.uniform(0, 0)


def random_init(words_num, dim):
    result = []
    for i in range(words_num):
        emd = [round(random_emb_value(), 5) for _ in range(dim)]
        result.append(emd)
    return np.array(result)


def process_embedding(embed_path, vocab_list, emb_size, out_path):
    result = []
    logging.info('word list size = {a}'.format(a=len(vocab_list)))
    word_embed_dict = load_embeddings(embed_path, emb_size, vocab_list)

    cnt = 0
    for index, word in enumerate(vocab_list):
        if index % 10 == 0:logging.info('process {a} words ...'.format(a=index))
        if word in word_embed_dict:
            emd = word_embed_dict[word]
        else:
            emd = [str(round(random_emb_value(), 5)) for _ in range(emb_size)]
            cnt += 1
        result.append(emd)

    logging.info('word size = {a}, {b} words have not  embedding value'.format(a=len(vocab_list), b=cnt))
    np.save(out_path, np.array(result))


def bin2txt(bin_file, txt_file):
    from gensim.models.keyedvectors import KeyedVectors
    model = KeyedVectors.load_word2vec_format(bin_file, binary=True)
    model.save_word2vec_format(txt_file, binary=False)


def generate_word_embeddings(bin_emb_path, txt_emb_path, word_list, word_embedding_file, emb_size=200):
    bin2txt(bin_emb_path, txt_emb_path)
    process_embedding(txt_emb_path, word_list, emb_size, word_embedding_file)


if __name__ == '__main__':
    pass