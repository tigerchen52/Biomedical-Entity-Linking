import numpy as np


def repre_word(words, embedding):
    '''
    to use embeddings to represent texts
    :param words: the input text, e.g., ['type', 'two', 'diabetes']
    :param embedding: a list of word embeddings
    :return:
    '''
    repre = []
    for w in words:
        temp = embedding[w]
        repre.append(temp)
    return repre


def cosine_similarity(vector1, vector2):
    '''
    a normal calculation of the cosine similarity
    '''
    return np.dot(vector1, vector2) / (1e-8 + np.linalg.norm(vector1) * (np.linalg.norm(vector2)))


def aligned_cos_sim(sentence1, sentence2):
    '''
    this function corresponds the equation (2) in our paper
    param sentence: two sequences of word embeddings
    '''

    def _cal(s1, s2):
        sum_sim = 0.0
        for i in range(len(s1)):
            word1 = s1[i]
            max_sim = 0.0
            for j in range(len(s2)):
                word2 = s2[j]
                sim = cosine_similarity(word1, word2)
                if sim > max_sim:
                    max_sim = sim
            sum_sim += max_sim
        return sum_sim

    sim1 = _cal(sentence1, sentence2)
    sim2 = _cal(sentence2, sentence1)

    aligned_sim = (sim1 + sim2) / (len(sentence1) + len(sentence2))
    return aligned_sim


def find_topk_candidates(mention, entity_set, emb_matrix, topk):
    '''
    param mention: a recognized mention in a given doc
    param entity_set: the entity set E that contains the surface form of each entity
    param emb_matrix: the pre-trained word embeddings
    param topk: the number of candidates
    return: topk candidates from the entity set
    '''
    men_repre = repre_word(mention, emb_matrix)
    entity_repre = dict([(e, aligned_cos_sim(men_repre, repre_word(e, emb_matrix))) for e in entity_set])
    result = sorted(entity_repre.items(), key=lambda e: e[1], reverse=True)
    return result[:topk]

