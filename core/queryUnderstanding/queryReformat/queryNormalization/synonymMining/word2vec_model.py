import numpy as np
import os
from core.tools import setup_logger
from config import get_cfg

logger = setup_logger()
cfg = get_cfg()


def synonym_detect(input_word_code_dict, top_k):
    logger.info('start w2v synonym detect......')
    cal_sim(cfg.BASE.ROUGH_WORD2VEC, input_word_code_dict, top_k)
    logger.info('w2v done!!!')


def load_embedding(embed_path, input_word_code_dict):
    word_list = []
    word_embed = []
    input_word_ids = dict()
    with open(embed_path, encoding='utf8') as f:
        line = f.readline()
        line = f.readline()
        index = 0
        while line:
            row = line.strip().split(' ')
            word_list.append(row[0])
            if row[0] in input_word_code_dict:
                input_word_ids[index] = row[0]
            embed = [float(e) for e in row[1:]]
            word_embed.append(embed)
            line = f.readline()
            index += 1
    return word_list, np.array(word_embed), input_word_ids


def cal_sim(path, input_word_code_dict, top_k):
    word_list, word_embed, input_word_ids = load_embedding(
        path, input_word_code_dict)
    l2_word_embed = np.sqrt(np.sum(np.square(word_embed), axis=1))
    normal_word_embed = np.array(
        [word_embed[i] / l2_word_embed[i] for i in range(len(word_embed))])
    input_word_embed = []
    input_word_list = []
    for index, word in input_word_ids.items():
        temp_embed = normal_word_embed[index]
        input_word_embed.append(temp_embed)
        input_word_list.append(word)
    input_word_embed = np.array(input_word_embed)
    normal_word_embed_T = normal_word_embed.T
    cos = np.matmul(input_word_embed, normal_word_embed_T)
    sorted_id = (-cos).argsort()
    line = ''
    for i, word in enumerate(input_word_list):
        code = input_word_code_dict[word]
        near_id = sorted_id[i][:top_k]
        nearst_word = [word_list[x] for x in near_id]
        line += str(code) + '\t' + word + '\t' + '|'.join(nearst_word) + '\n'
    with open(os.path.join(cfg.QUERY_NORMALIZATION.SYNONYM_PATH, 'w2v_synonym.txt'),
              'a',
              encoding='utf8') as f:
        f.write(line)


def cal_sim_valid(path):
    word_list, word_embed, _ = load_embedding(path, {})
    l2_word_embed = np.sqrt(np.sum(np.square(word_embed), axis=1))
    normal_word_embed = np.array(
        [word_embed[i] / l2_word_embed[i] for i in range(len(word_embed))])
    normal_word_embed_T = normal_word_embed.T
    cos = np.matmul(normal_word_embed, normal_word_embed_T)
    sorted_id = (-cos).argsort()
    line = ''
    for i in range(len(sorted_id)):
        near_id = sorted_id[i][:20]
        nearst_word = [word_list[x] for x in near_id]
        line += ','.join(nearst_word) + '\n'
    with open('../temp/embed_valid.txt', 'w', encoding='utf8') as f:
        f.write(line)


if __name__ == "__main__":
    cal_sim_valid(path='../temp/w2v_embed_300.bin')