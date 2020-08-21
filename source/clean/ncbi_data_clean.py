import pandas as pd
import re
import os
import logging
import random
import numpy as np
from clean.ncbi_load_data import load_entity, load_mention_of_each_entity, load_number_mapping, load_train_data
from nltk import word_tokenize

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


def get_number(text):
    result = re.findall(r"\d+\.?\d*", text)
    return result


def split_number_from_string(text):
    numbers = get_number(text)

    if len(numbers) == 0:return text

    number_dict = load_number_mapping()

    def get_mapping(x):
        for k, v in number_dict.items():
            if x in v:
                return k
        return None

    for num in numbers:
        mapping = get_mapping(num)
        if not mapping:continue
        text = text.replace(num, ' ' + mapping + ' ')

    text = ' '.join(text.split())

    return text


def preprocess(text):
    # lower
    text = str.lower(text)

    # punctuation
    text = remove_punctuation(text)

    # token
    token_words = word_tokenize(text)

    # number 2
    number_tokens = token_words[:]
    number_dict = load_number_mapping()
    for k, numbers in number_dict.items():
        for num in numbers:
            if num in number_tokens:
                index = number_tokens.index(num)
                number_tokens[index] = k

    replaced_number_tokens = []
    for token in number_tokens:
        rep_token = split_number_from_string(token)
        replaced_number_tokens.append(rep_token)

    text = ' '.join(replaced_number_tokens).strip()

    return text


def read_dictionary_from_KB(input_path, out_path, train_data_path):
    data = pd.read_csv(input_path, encoding='utf8')
    data.info()
    #df = data[['DiseaseName', 'DiseaseID', 'AltDiseaseIDs', 'Synonyms']]
    line = ''
    for index, row in data.iterrows():
        DiseaseName = row['DiseaseName']

        DiseaseName = preprocess(DiseaseName)

        if row['DiseaseID'][0] == 'M':
            DiseaseID = row['DiseaseID'][5:]
        else:
            DiseaseID = row['DiseaseID']
        line += DiseaseID + '\t' + DiseaseName
        AltDiseaseIDs = row['AltDiseaseIDs']
        if pd.isnull(AltDiseaseIDs):
            AltDiseaseIDs = ''
        line += '\t' + AltDiseaseIDs + '\n'

        #mention
        mention_of_entity = None
        if train_data_path != '':
            mention_of_entity = load_mention_of_each_entity(train_data_path)
        mention_list = None
        if DiseaseID in mention_of_entity:
            mention_list = mention_of_entity[DiseaseID]
            for men in mention_list:
                if men == DiseaseName:continue
                line += DiseaseID + '\t' + men + '\t' + AltDiseaseIDs + '\n'

        Synonyms = row['Synonyms']
        if not pd.isnull(Synonyms):
            synonym_list = Synonyms.split('|')
            for synonym in synonym_list:
                if mention_list and synonym in mention_list:continue
                synonym = preprocess(synonym)
                line += DiseaseID + '\t' + str.lower(synonym) + '\t' + AltDiseaseIDs + '\n'

    with open(out_path, 'w', encoding='utf8')as f:
        f.write(line)


def read_NCBI_data(input_path, entity_path, output_path, context_path):
    content = ''
    _, id_map = load_entity(entity_path)
    context = ''
    mention_context = ''
    cnt = 0
    with open(input_path, encoding='utf8')as f:
        line = f.readline()
        abb_dict = {}
        while line:
            if line == '\n':
                abb_dict = {}
                context = ''
            if line[:2] == '  ':
                row = line.strip().split('|')
                abb, expansion = row[0], row[-2]
                abb_dict[abb] = expansion

            row = line.split('\t')
            if len(row) != 6:
                if '|' in line:
                    row = line.split('|')
                    if row[1] == 't' or row[1] == 'a':
                        context += row[2]

                line = f.readline()
                continue

            cnt += 1
            if cnt % 10 == 0:
                print('process {a} line ...'.format(a=cnt))
            ID = row[0].strip()
            mention_name = row[3].strip()
            exp_name = mention_name
            # abbreviation expansion
            abb_tuple = sorted(abb_dict.items(), key=lambda e:len(e[0]), reverse=True)
            for k, v in abb_tuple:
                if k in exp_name:
                    exp_name = exp_name.replace(k, v)
            #lower
            mention_name = str.lower(mention_name)

            exp_name = preprocess(exp_name)

            #replace alt id
            entity_id = row[5].strip()
            if entity_id[0] == 'M':
                entity_id = entity_id[5:]

            if entity_id in id_map:
                entity_id = id_map[entity_id]

            content += ID + '\t' + mention_name + '\t' + entity_id + '\t' + exp_name + '\n'

            #context
            start = int(row[1]) - 1
            end = int(row[2]) + 1
            next_char = context[end]
            next_part = ''
            while next_char != '.' and next_char != '\n':
                next_part += next_char
                end += 1
                next_char = context[end]
            last_char = context[start]
            last_part = ''
            while last_char != '.' and last_char != '\n' and start >= 0:
                last_part = last_char + last_part
                start -= 1
                last_char = context[start]
            sentence = last_part + next_part
            sentence = preprocess(sentence)
            if len(sentence) == 0:
                sentence = exp_name
            mention_context += ID + '\t' + exp_name + '\t' + sentence + '\n'

            line = f.readline()

    with open(context_path, 'w', encoding='utf8')as f:
        f.write(mention_context)

    with open(output_path, 'w', encoding='utf8')as f:
        f.write(content)


def merge_data(train_path, develop_path, test_path, all_data_path):
    with open(train_path, encoding='utf8')as f:
        train_lines = f.readlines()
    with open(develop_path, encoding='utf8')as f:
        develop_lines = f.readlines()
    with open(test_path, encoding='utf8')as f:
        test_lines = f.readlines()
    all_data = train_lines + develop_lines + test_lines

    with open(all_data_path, 'w', encoding='utf8')as f:
        f.write(all_data)


def gen_mention_entity_prior(train_data_path, prior_path):
    _, all_data = load_train_data(train_data_path)
    mention_lable_dict = dict()
    for doc_id, mention, label in all_data:
        if '|' in label:continue
        if mention not in mention_lable_dict:
            mention_lable_dict[mention] = dict()

        if label not in mention_lable_dict[mention]:
            mention_lable_dict[mention][label] = 0
        mention_lable_dict[mention][label] += 1

    w_l = ''
    for mention in mention_lable_dict.keys():
        for label, cnt in mention_lable_dict[mention].items():
            w_l += mention + '\t' + label + '\t' + str(cnt) + '\n'

    with open(prior_path, 'w', encoding='utf8')as f:
        f.write(w_l)


def generate_vocabulary(word_vocab_path, all_data_path, entity_path, train_context_path, develop_context_path, test_context_path):
    vocabulary = set()
    with open(all_data_path, encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            row = line.strip().split('\t')
            for w in word_tokenize(row[3]):
                vocabulary.add(w)

    with open(entity_path, encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            row = line.strip().split('\t')
            for w in word_tokenize(row[1]):
                vocabulary.add(w)

    with open(train_context_path, encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            row = line.strip().split('\t')
            for w in word_tokenize(row[2]):
                vocabulary.add(w)

    with open(develop_context_path, encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            row = line.strip().split('\t')
            for w in word_tokenize(row[2]):
                vocabulary.add(w)

    with open(test_context_path, encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            row = line.strip().split('\t')
            for w in word_tokenize(row[2]):
                vocabulary.add(w)

    logging.info('word vocabulary size = {a}'.format(a=len(vocabulary)))
    w_l = ''
    for index, word in enumerate(vocabulary):
        w_l += str(index + 1) + '\t' + word + '\n'

    with open(word_vocab_path, 'w', encoding='utf8')as f:
        f.write(w_l)


def gen_char_vocabulary(char_vocab_path, all_data_path, entity_path):
    vocabulary = set()
    with open(all_data_path, encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            row = line.strip()
            for w in row:
                w = str.lower(w)
                if w not in vocabulary:
                    vocabulary.add(w)

    with open(entity_path, encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            row = line.strip()
            for w in row:
                w = str.lower(w)
                if w not in vocabulary:
                    vocabulary.add(w)

    logging.info('character vocabulary size = {a}'.format(a=len(vocabulary)))
    w_l = ''
    for index, word in enumerate(vocabulary):
        w_l += str(index + 1) + '\t' + word + '\n'

    with open(char_vocab_path, 'w', encoding='utf8')as f:
        f.write(w_l)


def clean_data(
        entity_in_path, entity_out_path, train_in_path, train_out_path, train_context_out,
        develop_in_path, develop_out_path, develop_context_out,
        test_in_path, test_out_path, test_context_out,
        word_vocab_path, all_data_path, char_vocab_path,
        prior_path
        ):

    logger.info('start to generate dataset... ')
    read_dictionary_from_KB(entity_in_path, entity_out_path, train_data_path='')
    read_NCBI_data(train_in_path, entity_out_path, train_out_path, train_context_out)
    read_dictionary_from_KB(entity_in_path, entity_out_path, train_out_path)
    read_NCBI_data(develop_in_path, entity_out_path, develop_out_path, develop_context_out)
    read_NCBI_data(test_in_path, entity_out_path, test_out_path, test_context_out)

    merge_data(train_out_path, develop_out_path, test_out_path, all_data_path)

    generate_vocabulary(word_vocab_path, all_data_path, entity_out_path, train_context_out, develop_context_out, test_context_out)
    gen_char_vocabulary(char_vocab_path, all_data_path, entity_out_path)

    gen_mention_entity_prior(train_out_path, prior_path)




if __name__ == '__main__':
    a = [1, 2]
    b = [3, 4]
    print(a+b)
