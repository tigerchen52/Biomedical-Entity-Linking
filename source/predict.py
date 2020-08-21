import numpy as np
from load_data import load_entity


def predict_batch(test_data, model, batch_size=None):
    result = model.predict(test_data, batch_size=batch_size)
    return result


def predict_data(test_data, entity_path, model, predict_path, score_path):
    entity_dict, id_map = load_entity(entity_path)
    acc_cnt, total_cnt = 0, 0
    w_l = ''
    all_score = ''
    for data, labels, raw_data in test_data:
        total_cnt += 1
        groud_truth, doc_id, mention = raw_data[0], raw_data[1], raw_data[2]

        raw_entity_list = data['entity_name']
        result = predict_batch(data, model, batch_size=len(labels))
        flatten_result = []
        for i in result:
            for j in i:
                flatten_result.append(j)
        pred_index = np.argmax(flatten_result)
        pred_label = labels[pred_index]
        pred_entity_name = raw_entity_list[pred_index]

        #all score
        all_score += doc_id + '\t' + mention
        for index, score in enumerate(flatten_result):
            entity_id = labels[index]
            entity_name = raw_entity_list[index]
            all_score += '\t' + entity_id + '\t' + entity_name + '\t' + str(round(score, 4))
        all_score += '\n'

        if pred_label == groud_truth:
            acc_cnt += 1
        else:
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
    print('total = {b}, acc = {c}, accuracy of the model is {a}'.format(a=accuracy, b=total_cnt, c=acc_cnt))
    with open(predict_path, 'w', encoding='utf8')as f:
        f.write(w_l)

    with open(score_path, 'w', encoding='utf8')as f:
        f.write(all_score)

    return accuracy


if __name__ == '__main__':
    flag = 1