from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
from transformers import BertTokenizerFast
import argparse
import pandas as pd
import pickle
import os
from tqdm import tqdm
import logging
import numpy as np
from qa.tools import setup_logger
from config import get_cfg

logger = setup_logger()


def preprocess():
    """
    对原始语料进行tokenize，将每段对话处理成如下形式："[CLS]utterance1[SEP]utterance2[SEP]utterance3[SEP]"
    """
    # 设置参数
    cfg = get_cfg()
    save_path = os.path.join(cfg.QUERY_SUGGESTION.GPT.SAVE_PATH, 'train.pkl')
    # 初始化tokenizer
    tokenizer = BertTokenizerFast(
        vocab_file='qa/queryUnderstanding/querySuggest/GPT2/vocab.txt',
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]")
    sep_id = tokenizer.sep_token_id
    cls_id = tokenizer.cls_token_id
    logger.info("preprocessing data,data path:{}, save path:{}".format(
        cfg.QUERY_SUGGESTION.GPT.TRAIN_DATA, save_path))

    # 读取训练数据集
    data = pd.read_csv(cfg.QUERY_SUGGESTION.GPT.TRAIN_DATA)

    logger.info("there are {} dialogue in dataset".format(data.shape[0]))

    # 开始进行tokenize
    # 保存所有的对话数据,每条数据的格式为："[CLS]utterance1[SEP]utterance2[SEP]utterance3[SEP]"
    dialogue_len = []  # 记录所有对话tokenize之后的长度，用于统计中位数与均值
    dialogue_list = []
    for index, row in tqdm(data.iterrows()):
        input_ids = [cls_id]
        input_ids += tokenizer.encode(row['source'], add_special_tokens=False)
        input_ids.append(sep_id)  # 每个utterance之后添加[SEP]，表示utterance结束
        input_ids += tokenizer.encode(row['target'], add_special_tokens=False)
        dialogue_len.append(len(input_ids))
        dialogue_list.append(input_ids)
    len_mean = np.mean(dialogue_len)
    len_median = np.median(dialogue_len)
    len_max = np.max(dialogue_len)
    with open(save_path, "wb") as f:
        pickle.dump(dialogue_list, f)
    logger.info("finish preprocessing data,the result is stored in {}".format(
            save_path))
    logger.info(
        "mean of dialogue len:{},median of dialogue len:{},max len:{}".format(
            len_mean, len_median, len_max))


if __name__ == '__main__':
    preprocess()