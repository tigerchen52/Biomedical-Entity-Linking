def load_candidates(can_path):
    candidate_dict = dict()
    with open(can_path, encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            row = line.strip().split('\t')
            mention = row[0]
            cans = row[1:]
            if mention not in candidate_dict:
                candidate_dict[mention] = list()
            for can in cans:
                can = can.strip().split('__')
                c_id, c_name = can[0], can[1]
                candidate_dict[mention].append((c_id, c_name, can[2]))
    return candidate_dict


def clean(path, outpath):
    w_l = ''
    can_dict = load_candidates(path)
    for mention, can_list in can_dict.items():
        can_list = can_list[:30]
        can_list = ['__'.join(can) for can in can_list]
        w_l += mention + '\t' + '\t'.join(can_list) + '\n'
    with open(outpath, 'w', encoding='utf8')as f:
        f.write(w_l)


if __name__ == '__main__':
    path = '../output/ncbi/candidates/test_candidates.txt'
    out_path = '../output/ncbi/candidates/test_candidates2.txt'
    clean(path, out_path)