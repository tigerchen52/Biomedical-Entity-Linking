from multiprocessing import Pool
from load.ncbi_load_data import load_vocabulary, load_entity_by_id, load_data_by_id, tran2ids, \
    load_train_data, load_entity, load_candidates
import numpy as np
import os


def get_candidate_dict(path, alpha, topk):
    candidate_dict = dict()
    raw_candidate_dict = dict()
    filter_cnt = 0
    with open(path, encoding='utf8')as f:
        lines = f.readlines()
        for line in lines:
            row = line.strip().split('\t')
            mention = row[0]
            candidate_dict[mention] = []
            raw_candidate_dict[mention] = []
            for candidates in row[1:topk+1]:
                candidates = candidates.split('__')
                score = 2.0
                d_id, name = candidates[0], candidates[1]
                if score > alpha:
                    raw_candidate_dict[mention].append((d_id, name, score))
                    candidate_dict[mention].append(d_id)
                else:
                    filter_cnt += 1

    print('filter candidates = {a}'.format(a=filter_cnt))
    return candidate_dict, raw_candidate_dict


def get_recall(path, alpha=0.0, topk=100):
    test_data, all_data = load_train_data('../output/test_data.txt')
    entity_dict, id_map = load_entity()
    cnt, total = 0, len(all_data)
    missing_set = dict()
    candidate_dict, raw_candidate_dict = get_candidate_dict(path, alpha, topk)

    for doc_id, mention, label in all_data:
        if mention not in candidate_dict:
            print('########' + mention)
            continue
        can_labels = candidate_dict[mention]
        if label in can_labels:
            cnt += 1
        else:
            if str.lower(label) == 'cui-less':
                entity = 'cui-less'
                cnt += 1
            elif label not in entity_dict:
                entity = 'miss_entity'
            else:
                entity = entity_dict[label][0]

            missing_content = mention + '\t' + label + '\t' + entity
            if missing_content not in missing_set:
                missing_set[missing_content] = 0
            missing_set[missing_content] += 1

    missing_set = sorted(missing_set.items(), key=lambda e:e[1], reverse=True)
    w_l = ''
    unmap_cnt = 0
    for (content, value) in missing_set:
        w_l += content + '\t' + str(value) + '\n'
        if 'cui-less' in content or 'miss_entity' in content:
            unmap_cnt += value
    print('unmapped mention = {a}'.format(a=unmap_cnt))
    with open('../output/candidates/missing_candidates.txt', 'w', encoding='utf8')as f:
        f.write(w_l)

    print('total = {a}, find {b}, rate = {c}'.format(a=total, b=cnt, c=cnt * 1.0 / total))
    return cnt * 1.0 / total


def repre_word(word_ids, embedding):
    repre = []
    for w_id in word_ids:
        temp = embedding[w_id]
        repre.append(temp)
    return repre


def cal_cosine_similarity(vector1, vector2):
    return np.dot(vector1, vector2)/(1e-8 + np.linalg.norm(vector1)*(np.linalg.norm(vector2)))


def aligned_cos_sim(sentence1, sentence2):
    def _cal(s1, s2):
        sum_sim = 0.0
        for i in range(len(s1)):
            word1 = s1[i]
            max_sim = 0.0
            for j in range(len(s2)):
                word2 = s2[j]
                sim = cal_cosine_similarity(word1, word2)
                if sim > max_sim:
                    max_sim = sim
            sum_sim += max_sim
        return sum_sim

    sim1 = _cal(sentence1, sentence2)
    sim2 = _cal(sentence2, sentence1)

    final_sim = (sim1 + sim2) / (len(sentence1) + len(sentence2))
    return final_sim


def find_can_by_cos(mention, mention_id, raw_entity_dict, word_embeddings, topk):
    candidate_dict = {}
    mention_repre = repre_word(mention_id, word_embeddings)
    for label, synonyms in raw_entity_dict.items():
        max_sim, max_entity = -1, ''
        for index in range(len(synonyms)):
            syn_id, synonym = synonyms[index][0], synonyms[index][1]
            if synonym == mention:
                max_sim = 1.2
                max_entity = synonym
                break
            else:
                syn_repre = repre_word(syn_id, word_embeddings)
                cos_sim = aligned_cos_sim(mention_repre, syn_repre)
                if cos_sim > max_sim:
                    max_sim = cos_sim
                    max_entity = synonym
        candidate_dict[label + '__' + max_entity] = max_sim

    result = sorted(candidate_dict.items(), key=lambda e: e[1], reverse=True)
    return result[:topk]


def generate_candidate_by_cos(data_path, outpath):
    max_len = 25
    top_k = 200
    entity_path = '../output/entity_kb.txt'
    embedding_file = '../output/embed/word2vec_200_dim.embeddings.npy'
    # load test data
    all_data, test_data = load_train_data(data_path)

    all_mentions = list(all_data.keys())
    print('the number of mentions = {a}'.format(a=len(all_mentions)))

    vocab_dict, _ = load_vocabulary('../output/vocabulary_origin.dict')
    embedding_matrix = np.load(embedding_file)
    embedding_matrix = embedding_matrix.astype(np.float)
    _, _, raw_entity_dict = load_entity_by_id(entity_path, max_len=max_len, char_max_len=25, use_pad=False)

    # candidate_dict = load_candidates(outpath)
    # f = open('../output/candidates/add_candidates.txt', 'w')
    # for mention in all_mentions:
    #     if mention in candidate_dict:continue
    #     print('mention = {b}.....'.format(b=mention))
    #     mention_id = tran2ids(mention, vocab_dict, max_len=max_len, use_pad=False)
    #     result = find_can_by_cos(mention, mention_id, raw_entity_dict, embedding_matrix, top_k)
    #     temp = [r[0] + '__' + str(round(r[1], 5)) for r in result]
    #     content = mention + '\t' + '\t'.join(temp) + '\n'
    #     f.write(content)
    #     f.flush()

    p = Pool(25)
    cnt, id_cnt = 0, 0
    temp_list = []
    for mention in all_mentions:
        cnt += 1
        temp_list.append(mention)
        if cnt % 150 == 0:
            id_cnt += 1
            print('process {a} line, mention = {b}.....'.format(a=cnt, b=mention))
            args = (temp_list, vocab_dict, max_len, raw_entity_dict, embedding_matrix, top_k, id_cnt)
            p.apply_async(generate_candidate, args=(args,))
            temp_list = []
    args = (temp_list, vocab_dict, max_len, raw_entity_dict, embedding_matrix, top_k, id_cnt+1)
    p.apply_async(generate_candidate, args=(args,))

    p.close()
    p.join()


def generate_candidate(args):
    all_mentions, vocab_dict, max_len, raw_entity_dict, embedding_matrix, top_k, id_cnt = args
    outpath = '../output/candidates/temp/temp_{a}'.format(a=id_cnt)
    f = open(outpath, 'w', encoding='utf8')

    for mention in all_mentions:
        print('mention = {b}.....'.format(b=mention))
        mention_id = tran2ids(mention, vocab_dict, max_len=max_len, use_pad=False)
        result = find_can_by_cos(mention, mention_id, raw_entity_dict, embedding_matrix, top_k)
        temp = [r[0] + '__' + str(round(r[1], 5)) for r in result]
        content = mention + '\t' + '\t'.join(temp) + '\n'
        f.write(content)
        f.flush()


def get_all_file_name(path):
    file_list = list()
    g = os.walk(path)
    for path, dir_list, file_name_list in g:
        for file_name in file_name_list:
            file = os.path.join(path, file_name)
            file_list.append(file)
    return file_list


def merge_candidate_files(path):
    file_list = get_all_file_name(path)

    w_l = ''
    for file_name in file_list:
        with open(file_name)as f:
            content = f.read()
            w_l += content

    with open('../output/candidates/training_aligned_cos_with_mention_candidate.txt', 'w')as f:
        f.write(w_l)


def look_up_candidates_for_test_set(can_path, outpath):
    from load_data import load_candidates
    candidate_dict = load_candidates(can_path)
    data_dict, all_data = load_train_data('../output/test_data.txt')

    w_l = ''
    for mention, label in data_dict.items():
        candidates = candidate_dict[mention]
        temp = list()
        alpha = 0
        for index, (c_id, c_name, c_score) in enumerate(candidates):
            if index == 0 and c_score == 0.0:
                temp.append('cui_less' + '__' + 'cui_less' + '__' + str(c_score))
                break
            if index == 0 and c_score >= 1.0:
                alpha = 1.0
            if c_score >= alpha:
                temp.append(c_id+'__'+c_name+'__'+str(c_score))
        w_l += mention + '\t' + '\t'.join(temp) + '\n'

    with open(outpath, 'w', encoding='utf8')as f:
        f.write(w_l)


def merge_candidate(alpha1, alpha2, topk=100, topk1=10, topk2=10):
    path1 = '../output/candidates/test_candidates.txt'
    path2 = '../output/candidates/training_ed_mention_candidate.txt'
    candidate_dict1, raw_candidate_dict1 = get_candidate_dict(path1, alpha=alpha1, topk=topk1)
    candidate_dict2, raw_candidate_dict2 = get_candidate_dict(path2, alpha=alpha2, topk=topk2)

    w_l = ''
    for mention, can_list1 in raw_candidate_dict1.items():
        can_list2 = raw_candidate_dict2[mention]
        existed = [e[0] for e in can_list2]
        for d_id, name, score in can_list1:
            if score >= 1.0:
                can_list2 = list()
                can_list2.append((d_id, name, score))
                continue
            if d_id not in existed:
                can_list2.append((d_id, name, score))
        can_list2 = [e[0] + '__' + e[1] + '__' + str(e[2]) for e in can_list2[:topk]]

        w_l += mention + '\t' + '\t'.join(can_list2) + '\n'

    with open('../output/candidates/merge_candidates.txt', 'w', encoding='utf8')as f:
        f.write(w_l)

    acc = get_recall('../output/candidates/merge_candidates.txt', topk=topk)
    return acc


def find_close_mentions(mention, candidate, mention_doc_list, embedding_matrix, vocab_dict):
    candidate_id = tran2ids(candidate, vocab_dict, max_len=15, use_pad=False)
    cosine_dict = dict()
    for mention_in_doc in mention_doc_list:
        if mention == mention_in_doc: continue
        men_doc_id = tran2ids(mention_in_doc, vocab_dict, max_len=15, use_pad=False)
        can_repre = repre_word(candidate_id, embedding_matrix)
        men_repre = repre_word(men_doc_id, embedding_matrix)
        cosine = aligned_cos_sim(can_repre, men_repre)
        cosine_dict[mention_in_doc] = cosine

    sorted_result = sorted(cosine_dict.items(), key=lambda e:e[1], reverse=True)
    result = [m for m, s in sorted_result][:10]

    return result


if __name__ == '__main__':
    flag = 0
    if flag == 0:
        topk, alpha = 20, 0
        path = '../output/candidates/test_candidates.txt'
        get_recall(path, alpha, topk=topk)
    if flag == 1:
        max_acc, max_alpha = 0, 0
        path = '../output/candidates/test_candidates.txt'
        for i in range(40):
            alpha = 0.5 + i * 0.01
            print('alpha = {a} '.format(a=alpha))
            acc = get_recall(path, alpha, topk=20)
            if acc > max_acc:
                max_acc = acc
                max_alpha = alpha
        print('max_acc = {a}, params = {b}'.format(a=max_acc, b=max_alpha))
    if flag == 2:
        inpath = '../output/all_data.txt'
        outpath = '../output/candidates/training_aligned_cos_with_mention_candidate.txt'
        generate_candidate_by_cos(inpath, outpath)
        merge_candidate_files('../output/candidates/temp')
    if flag == 4:
        can_path = '../output/candidates/training_aligned_cos_with_mention_candidate.txt'
        outpath = '../output/candidates/test_candidates.txt'
        look_up_candidates_for_test_set(can_path, outpath)
    if flag == 5:
        alpha1_list = [0.4, 0.45, 0.5, 0.55]
        alpha2_list = [0.35, 0.40, 0.45, 0.5]
        max_acc, max_params = 0, None
        K = 20
        for i in range(1, K, 1):
            topk1 = i
            topk2 = K-i
            for alpha1 in alpha1_list:
                for alpha2 in alpha2_list:
                    print('K= {e}, alpha1 = {a}, alpha2 = {b}, topk1 = {c}, topk2 = {d}:'.
                          format(a=alpha1, b=alpha2, c=topk1, d=topk2, e=K))
                    acc = merge_candidate(alpha1, alpha2, topk=K, topk1=topk1, topk2=topk2)
                    if acc > max_acc:
                        max_acc = acc
                        max_params = (topk1, topk2, alpha1, alpha2)
        print('max_acc = {a}, params = {b}'.format(a=max_acc, b=max_params))
        #merge_candidate(alpha1=0.4, alpha2=0.35, topk=25, topk1=24, topk2=1)