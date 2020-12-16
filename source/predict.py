import numpy as np
from load_data import load_entity, load_candidates2, load_train_data


def predict_batch(test_data, model, batch_size=None):
    result = model.predict(test_data, batch_size=batch_size)
    return result


def predict_data(test_data, entity_path, model, predict_path, score_path, test_path, dataset):
    entity_dict, id_map = load_entity(entity_path)
    acc_cnt, total_cnt = 0, 0
    w_l = ''
    all_score = ''
    for data, labels, raw_data in test_data:
        total_cnt += 1
        groud_truth, doc_id, mention = raw_data[0], raw_data[1], raw_data[2]

        raw_entity_list = data['entity_name']
        pred_result = predict_batch(data, model, batch_size=len(labels))
        pred_result = [j for r in pred_result for j in r]
        pred_index = np.argmax(pred_result)
        pred_label = labels[pred_index]
        pred_entity_name = raw_entity_list[pred_index]

        #all score
        all_score += doc_id + '\t' + mention
        for index, score in enumerate(pred_result):
            entity_id = labels[index]
            entity_name = raw_entity_list[index]
            all_score += '\t' + entity_id + '\t' + entity_name + '\t' + str(round(score, 4))
        all_score += '\n'

        if pred_label == groud_truth:
            acc_cnt += 1
        else:
            # write wrong results down
            if groud_truth in id_map:
                groud_truth = id_map[groud_truth]

            ground_name = ''
            if '+' in groud_truth:
                ground_name = groud_truth
            else:
                if groud_truth not in entity_dict:
                    ground_name = ground_name
                else:
                    ground_name = entity_dict[groud_truth][0]
            w_l += doc_id + '\t' + mention + '\t' + groud_truth + '\t' + \
                   ground_name + '\t' + pred_label + '\t' + pred_entity_name + '\n'

    accuracy = 1.0 * acc_cnt / (total_cnt+1)
    with open(predict_path, 'w', encoding='utf8')as f:
        f.write(w_l)

    with open(score_path, 'w', encoding='utf8')as f:
        f.write(all_score)

    if dataset == 'clef':
        return post_predict(test_path, score_path, entity_path)
    else:
        return accuracy


def post_predict(test_path, score_path, entity_path, alpha=0.75):
    candidate_dict = load_candidates2(score_path)
    test_data, all_data = load_train_data(test_path)
    entity_dict, _ = load_entity(entity_path)

    acc_cnt, w_l = 0, ''

    predict_dict = dict()
    for mention, candidates in candidate_dict.items():
        if len(candidates) == 1:
            predict_dict[mention] = (candidates[0][0], candidates[0][1])
            continue
        max_score, max_can = candidates[0][2], candidates[0]
        for e_id, e_name, e_score in candidates:
            if e_score > max_score:
                max_score = e_score
                max_can = (e_id, e_name, e_score)

        e_id, e_name, e_score = max_can
        if e_score < alpha:
            e_id, e_name = 'cui-less', 'cui-less'
        predict_dict[mention] = (e_id, e_name)

    for doc_id, mention, label in all_data:
        if str.lower(label) == 'cui-less':
            label = 'cui-less'
        pred_label, pred_entity_name = predict_dict[mention]
        if pred_label == label:
            acc_cnt += 1
        else:
            entity_name = 'None'
            if label in entity_dict:
                entity_name = entity_dict[label][0]
            w_l += doc_id + '\t' + mention + '\t' + label + '\t' + \
                   entity_name + '\t' + pred_label + '\t' + pred_entity_name + '\n'

    with open('../checkpoints/post_predict_result.txt', 'w')as f:
        f.write(w_l)

    total_cnt = len(all_data)
    accuracy = 1.0 * acc_cnt / (total_cnt)
    return accuracy


if __name__ == '__main__':
    flag = 1