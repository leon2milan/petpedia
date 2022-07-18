import argparse
import math
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import logging
from datetime import datetime
import os
from torch.utils.data import Dataset, DataLoader
from os.path import join, exists
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torch.nn import DataParallel
import transformers
import pickle
import sys
from qa.tools.torch_utils import EarlyStopping
from sklearn.model_selection import train_test_split
from qa.queryUnderstanding.querySuggest.GPT2.data_parallel import BalancedDataParallel
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config
from transformers import BertTokenizerFast
import pandas as pd
import torch.nn.utils.rnn as rnn_utils
import numpy as np
from config import get_cfg
from qa.tools import setup_logger
from torch.utils.data import Dataset
import torch

logger = setup_logger()


class MyDataset(Dataset):
    """
    """
    def __init__(self, input_list, max_len):
        self.input_list = input_list
        self.max_len = max_len

    def __getitem__(self, index):
        input_ids = self.input_list[index]
        input_ids = input_ids[:self.max_len]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids

    def __len__(self):
        return len(self.input_list)


def collate_fn(batch):
    input_ids = rnn_utils.pad_sequence(batch,
                                       batch_first=True,
                                       padding_value=0)
    labels = rnn_utils.pad_sequence(batch,
                                    batch_first=True,
                                    padding_value=-100)
    return input_ids, labels


# def padding_batch(data_list, pad_id):
#     """
#     使用pad_id将data_list的每条数据，填充至data_list中最长的长度
#     :param data_list:
#     :param pad_id:
#     :return:
#     """
#     # 统计data_list中的最大长度
#     max_len = 0
#     for data in data_list:
#         max_len = max_len if max_len > len(data) else len(data)
#
#     # 对数据进行padding
#     new_data_list = []
#     for data in data_list:
#         new_data = data + [pad_id] * (max_len - len(data))
#         new_data_list.append(new_data)
#     return new_data_list


def load_dataset(logger, cfg):
    """
    加载训练集和验证集
    """
    logger.info("loading training dataset and validating dataset")
    train_path = os.path.join(cfg.QUERY_SUGGESTION.GPT.SAVE_PATH, 'train.pkl')

    with open(train_path, "rb") as f:
        input_list = pickle.load(f)

    # 划分训练集与验证集
    val_num = cfg.QUERY_SUGGESTION.VAL_NUM
    input_list_train = input_list[val_num:]
    input_list_val = input_list[:val_num]
    # test
    # input_list_train = input_list_train[:24]
    # input_list_val = input_list_val[:24]

    train_dataset = MyDataset(input_list_train,
                              cfg.QUERY_SUGGESTION.GPT.MAX_LEN)
    val_dataset = MyDataset(input_list_val, cfg.QUERY_SUGGESTION.GPT.MAX_LEN)

    return train_dataset, val_dataset


def train_epoch(model, train_dataloader, optimizer, scheduler, logger, epoch,
                cfg):
    model.train()
    device = torch.device(
        'cuda' if cfg.REPRESENTATION.SIMCSE.DEVICE else 'cpu')
    # pad_id = args.pad_id
    # sep_id = args.sep_id
    ignore_index = cfg.QUERY_SUGGESTION.GPT.IGNORE_INDEX
    epoch_start_time = datetime.now()
    total_loss = 0  # 记录下整个epoch的loss的总和

    # epoch_correct_num:每个epoch中,output预测正确的word的数量
    # epoch_total_num: 每个epoch中,output预测的word的总数量
    epoch_correct_num, epoch_total_num = 0, 0

    for batch_idx, (input_ids, labels) in enumerate(train_dataloader):
        # 捕获cuda out of memory exception
        try:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            outputs = model.forward(input_ids, labels=labels)
            logits = outputs.logits
            loss = outputs.loss
            loss = loss.mean()

            # 统计该batch的预测token的正确数与总数
            batch_correct_num, batch_total_num = calculate_acc(
                logits, labels, ignore_index=ignore_index)
            # 统计该epoch的预测token的正确数与总数
            epoch_correct_num += batch_correct_num
            epoch_total_num += batch_total_num
            # 计算该batch的accuracy
            batch_acc = batch_correct_num / batch_total_num

            total_loss += loss.item()
            if cfg.QUERY_SUGGESTION.GPT.GRADIENT_ACC_STEPS > 1:
                loss = loss / cfg.QUERY_SUGGESTION.GPT.GRADIENT_ACC_STEPS

            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           cfg.QUERY_SUGGESTION.WARMUP_STEPS)

            # 进行一定step的梯度累计之后，更新参数
            if (batch_idx +
                    1) % cfg.QUERY_SUGGESTION.GPT.GRADIENT_ACC_STEPS == 0:
                # 更新参数
                optimizer.step()
                # 更新学习率
                scheduler.step()
                # 清空梯度信息
                optimizer.zero_grad()

            if (batch_idx + 1) % cfg.QUERY_SUGGESTION.LOG_STEPS == 0:
                logger.info(
                    "batch {} of epoch {}, loss {}, batch_acc {}, lr {}".
                    format(
                        batch_idx + 1, epoch + 1,
                        loss.item() *
                        cfg.QUERY_SUGGESTION.GPT.GRADIENT_ACC_STEPS, batch_acc,
                        scheduler.get_lr()))

            del input_ids, outputs

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                logger.info("WARNING: ran out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                logger.info(str(exception))
                raise exception

    # 记录当前epoch的平均loss与accuracy
    epoch_mean_loss = total_loss / len(train_dataloader)
    epoch_mean_acc = epoch_correct_num / epoch_total_num
    logger.info("epoch {}: loss {}, predict_acc {}".format(
        epoch + 1, epoch_mean_loss, epoch_mean_acc))

    # save model
    logger.info('saving model for epoch {}'.format(epoch + 1))
    model_path = join(cfg.QUERY_SUGGESTION.GPT.SAVE_PATH,
                      'epoch{}'.format(epoch + 1))
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(model_path)
    logger.info('epoch {} finished'.format(epoch + 1))
    epoch_finish_time = datetime.now()
    logger.info('time for one epoch: {}'.format(epoch_finish_time -
                                                epoch_start_time))

    return epoch_mean_loss


def validate_epoch(model, validate_dataloader, logger, epoch, cfg):
    logger.info("start validating")
    model.eval()
    device = torch.device(
        'cuda' if cfg.REPRESENTATION.SIMCSE.DEVICE else 'cpu')
    # pad_id = args.pad_id
    # sep_id = args.sep_id
    ignore_index = cfg.QUERY_SUGGESTION.GPT.IGNORE_INDEX
    epoch_start_time = datetime.now()
    total_loss = 0
    # 捕获cuda out of memory exception
    try:
        with torch.no_grad():
            for batch_idx, (input_ids,
                            labels) in enumerate(validate_dataloader):
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                outputs = model.forward(input_ids, labels=labels)
                logits = outputs.logits
                loss = outputs.loss
                loss = loss.mean()

                total_loss += loss.item()
                del input_ids, outputs

            # 记录当前epoch的平均loss
            epoch_mean_loss = total_loss / len(validate_dataloader)
            logger.info("validate epoch {}: loss {}".format(
                epoch + 1, epoch_mean_loss))
            epoch_finish_time = datetime.now()
            logger.info(
                'time for validating one epoch: {}'.format(epoch_finish_time -
                                                           epoch_start_time))
            return epoch_mean_loss
    except RuntimeError as exception:
        if "out of memory" in str(exception):
            logger.info("WARNING: ran out of memory")
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        else:
            logger.info(str(exception))
            raise exception


def train(model, logger, train_dataset, validate_dataset, cfg):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.QUERY_SUGGESTION.GPT.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True)
    validate_dataloader = DataLoader(
        validate_dataset,
        batch_size=cfg.QUERY_SUGGESTION.GPT.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True)
    early_stopping = EarlyStopping(
        cfg.QUERY_SUGGESTION.PATIENT,
        verbose=True,
        save_path=cfg.QUERY_SUGGESTION.GPT.SAVE_PATH)
    t_total = len(
        train_dataloader
    ) // (cfg.QUERY_SUGGESTION.GPT.GRADIENT_ACC_STEPS * cfg.QUERY_SUGGESTION.GPT.EPOCH + 1)
    optimizer = transformers.AdamW(model.parameters(),
                                   lr=cfg.QUERY_SUGGESTION.GPT.LR,
                                   eps=cfg.QUERY_SUGGESTION.GPT.EPS)
    # scheduler = transformers.WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.QUERY_SUGGESTION.WARMUP_STEPS,
        num_training_steps=t_total)

    logger.info('starting training')

    # 用于记录每个epoch训练和验证的loss
    train_losses, validate_losses = [], []
    # 记录验证集的最小loss
    best_val_loss = 10000
    # 开始训练
    for epoch in range(cfg.QUERY_SUGGESTION.GPT.EPOCH):
        # ========== train ========== #
        train_loss = train_epoch(model=model,
                                 train_dataloader=train_dataloader,
                                 optimizer=optimizer,
                                 scheduler=scheduler,
                                 logger=logger,
                                 epoch=epoch,
                                 cfg=cfg)
        train_losses.append(train_loss)

        # ========== validate ========== #
        validate_loss = validate_epoch(model=model,
                                       validate_dataloader=validate_dataloader,
                                       logger=logger,
                                       epoch=epoch,
                                       cfg=cfg)
        validate_losses.append(validate_loss)

        # 保存当前困惑度最低的模型，困惑度低，模型的生成效果不一定会越好
        if validate_loss < best_val_loss:
            best_val_loss = validate_loss
            logger.info('saving current best model for epoch {}'.format(epoch +
                                                                        1))
            model_path = join(cfg.QUERY_SUGGESTION.GPT.SAVE_PATH,
                              'min_ppl_model'.format(epoch + 1))
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(model_path)

        #  如果patience=0,则不进行early stopping
        if cfg.QUERY_SUGGESTION.PATIENT == 0:
            continue
        early_stopping(validate_loss, model)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break
    logger.info('training finished')
    logger.info("train_losses:{}".format(train_losses))
    logger.info("validate_losses:{}".format(validate_losses))


def caculate_loss(logit, target, pad_idx, smoothing=True):
    if smoothing:
        logit = logit[..., :-1, :].contiguous().view(-1, logit.size(2))
        target = target[..., 1:].contiguous().view(-1)

        eps = 0.1
        n_class = logit.size(-1)

        one_hot = torch.zeros_like(logit).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(logit, dim=1)

        non_pad_mask = target.ne(pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).mean()  # average later
    else:
        # loss = F.cross_entropy(predict_logit, target, ignore_index=pad_idx)
        logit = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))
        labels = target[..., 1:].contiguous().view(-1)
        loss = F.cross_entropy(logit, labels, ignore_index=pad_idx)
    return loss


def calculate_acc(logit, labels, ignore_index=-100):
    logit = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))
    labels = labels[..., 1:].contiguous().view(-1)

    _, logit = logit.max(dim=-1)  # 对于每条数据，返回最大的index
    # 进行非运算，返回一个tensor，若labels的第i个位置为pad_id，则置为0，否则为1
    non_pad_mask = labels.ne(ignore_index)
    n_correct = logit.eq(labels).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    return n_correct, n_word


def main():
    # 初始化参数
    cfg = get_cfg()

    device = torch.device(
        'cuda' if cfg.REPRESENTATION.SIMCSE.DEVICE else 'cpu')

    if cfg.QUERY_SUGGESTION.GPT.BATCH_SIZE < 2048 and cfg.QUERY_SUGGESTION.WARMUP_STEPS <= 4000:
        print('[Warning] The warmup steps may be not enough.\n' \
              '(sz_b, warmup) = (2048, 4000) is the official setting.\n' \
              'Using smaller batch w/o longer warmup may cause ' \
              'the warmup stage ends with only little data trained.')

    # 初始化tokenizer
    tokenizer = BertTokenizerFast(
        vocab_file='qa/queryUnderstanding/querySuggest/GPT2/vocab.txt',
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]")
    # args.sep_id = tokenizer.sep_token_id
    # args.pad_id = tokenizer.pad_token_id
    # args.cls_id = tokenizer.cls_token_id

    # 创建模型的输出目录
    if not os.path.exists(cfg.QUERY_SUGGESTION.GPT.SAVE_PATH):
        os.mkdir(cfg.QUERY_SUGGESTION.GPT.SAVE_PATH)

    # 创建模型
    if cfg.QUERY_SUGGESTION.PRETRAINED_MODEL:  # 加载预训练模型
        model = GPT2LMHeadModel.from_pretrained(cfg.QUERY_SUGGESTION.PRETRAINED_MODEL)
    else:  # 初始化模型
        model_config = GPT2Config.from_json_file('qa/queryUnderstanding/querySuggest/GPT2/config.json')
        model = GPT2LMHeadModel(config=model_config)
    model = model.to(device)
    logger.info('model config:\n{}'.format(model.config.to_json_string()))
    assert model.config.vocab_size == tokenizer.vocab_size

    # 并行训练模型
    if cfg.REPRESENTATION.SIMCSE.DEVICE and torch.cuda.device_count() > 1:
        model = DataParallel(model).cuda()

    # 计算模型参数数量
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    logger.info('number of model parameters: {}'.format(num_parameters))

    # 加载训练集和验证集
    # ========= Loading Dataset ========= #
    train_dataset, validate_dataset = load_dataset(logger, cfg)

    train(model, logger, train_dataset, validate_dataset, cfg)


if __name__ == '__main__':
    main()