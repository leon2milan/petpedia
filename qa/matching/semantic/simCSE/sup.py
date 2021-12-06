# -*- encoding: utf-8 -*-

import random
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer
from qa.tools import setup_logger

from config.config import get_cfg

logger = setup_logger()


def load_data(name: str, path: str) -> List:
    """根据名字加载不同的数据集
    """
    with open(path, 'r', encoding='utf8') as f:
        return [(line.split("||")[1], line.split("||")[2], line.split("||")[3])
                for line in f]


class TrainDataset(Dataset):
    """训练数据集, 重写__getitem__和__len__方法
    """
    def __init__(self, cfg, data: List):
        self.cfg = cfg
        self.data = data

    def __len__(self):
        return len(self.data)

    def text_2_id(self, text: str):
        return tokenizer([text[0], text[1], text[2]],
                         max_length=self.cfg.REPRESENTATION.SIMCSE.MAXLEN,
                         truncation=True,
                         padding='max_length',
                         return_tensors='pt')

    def __getitem__(self, index: int):
        return self.text_2_id(self.data[index])


class TestDataset(Dataset):
    """测试数据集, 重写__getitem__和__len__方法
    """
    def __init__(self, cfg, data: List):
        self.cfg = cfg
        self.data = data

    def __len__(self):
        return len(self.data)

    def text_2_id(self, text: str):
        return tokenizer(text,
                         max_length=self.cfg.REPRESENTATION.SIMCSE.MAXLEN,
                         truncation=True,
                         padding='max_length',
                         return_tensors='pt')

    def __getitem__(self, index):
        line = self.data[index]
        return self.text_2_id([line[0]]), self.text_2_id([line[1]
                                                          ]), int(line[2])


class SimcseModel(nn.Module):
    """Simcse有监督模型定义"""
    def __init__(self, cfg):
        super(SimcseModel, self).__init__()
        self.cfg = cfg
        self.bert = BertModel.from_pretrained(
            self.cfg.REPRESENTATION.SIMCSE.PRETRAINED_MODEL.PRETRAINED_MODEL)
        self.pooling = self.cfg.REPRESENTATION.SIMCSE.PRETRAINED_MODEL.POOLING

    def forward(self, input_ids, attention_mask, token_type_ids):

        # out = self.bert(input_ids, attention_mask, token_type_ids)
        out = self.bert(input_ids,
                        attention_mask,
                        token_type_ids,
                        output_hidden_states=True)

        if self.pooling == 'cls':
            return out.last_hidden_state[:, 0]  # [batch, 768]

        if self.pooling == 'pooler':
            return out.pooler_output  # [batch, 768]

        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1,
                                                   2)  # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(
                -1)  # [batch, 768]

        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1,
                                                   2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1,
                                                   2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first,
                                         kernel_size=last.shape[-1]).squeeze(
                                             -1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last,
                                        kernel_size=last.shape[-1]).squeeze(
                                            -1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)),
                            dim=1)  # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2),
                                    kernel_size=2).squeeze(-1)  # [batch, 768]


def simcse_sup_loss(y_pred: 'tensor', device) -> 'tensor':
    """有监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 3, 768]
    
    """
    # 得到y_pred对应的label, 每第三句没有label, 跳过, label= [1, 0, 4, 3, ...]
    y_true = torch.arange(y_pred.shape[0], device=device)
    use_row = torch.where((y_true + 1) % 3 != 0)[0]
    y_true = (use_row - use_row % 3 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
    # 选取有效的行
    sim = torch.index_select(sim, 0, use_row)
    # 相似度矩阵除以温度系数
    sim = sim / 0.05
    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(sim, y_true)
    return torch.mean(loss)


def eval(model, dataloader, device) -> float:
    """模型评估函数 
    批量预测, 计算cos_sim, 转成numpy数组拼接起来, 一次性求spearman相关度
    """
    model.eval()
    sim_tensor = torch.tensor([], device=device)
    label_array = np.array([])
    with torch.no_grad():
        for source, target, label in dataloader:
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source['input_ids'].squeeze(1).to(device)
            source_attention_mask = source['attention_mask'].squeeze(1).to(
                device)
            source_token_type_ids = source['token_type_ids'].squeeze(1).to(
                device)
            source_pred = model(source_input_ids, source_attention_mask,
                                source_token_type_ids)
            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target['input_ids'].squeeze(1).to(device)
            target_attention_mask = target['attention_mask'].squeeze(1).to(
                device)
            target_token_type_ids = target['token_type_ids'].squeeze(1).to(
                device)
            target_pred = model(target_input_ids, target_attention_mask,
                                target_token_type_ids)
            # concat
            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            label_array = np.append(label_array, np.array(label))
    # corrcoef
    return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation


def train(model, train_dl, dev_dl, optimizer, device, save_path,
          batch_size) -> None:
    """模型训练函数 
    """
    model.train()
    global best
    early_stop_batch = 0
    for batch_idx, source in enumerate(tqdm(train_dl), start=1):
        # 维度转换 [batch, 3, seq_len] -> [batch * 3, sql_len]
        real_batch_num = source.get('input_ids').shape[0]
        input_ids = source.get('input_ids').view(real_batch_num * 3,
                                                 -1).to(device)
        attention_mask = source.get('attention_mask').view(
            real_batch_num * 3, -1).to(device)
        token_type_ids = source.get('token_type_ids').view(
            real_batch_num * 3, -1).to(device)
        # 训练
        out = model(input_ids, attention_mask, token_type_ids)
        loss = simcse_sup_loss(out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 评估
        if batch_idx % 10 == 0:
            logger.info(f'loss: {loss.item():.4f}')
            corrcoef = eval(model, dev_dl)
            model.train()
            if best < corrcoef:
                early_stop_batch = 0
                best = corrcoef
                torch.save(model.state_dict(), save_path)
                logger.info(
                    f"higher corrcoef: {best:.4f} in batch: {batch_idx}, save model"
                )
                continue
            early_stop_batch += 1
            if early_stop_batch == 10:
                logger.info(
                    f"corrcoef doesn't improve for {early_stop_batch} batch, early stop!"
                )
                logger.info(
                    f"train use sample number: {(batch_idx - 10) * batch_size}"
                )
                return


if __name__ == '__main__':

    cfg = get_cfg()
    device = cfg.REPRESENTATION.SIMCSE.DEVICE
    model_path = cfg.REPRESENTATION.SIMCSE.PRETRAINED_MODEL
    batch_size = cfg.REPRESENTATION.SIMCSE.BATCH_SIZE
    save_path = cfg.REPRESENTATION.SIMCSE.SAVE_PATH
    logger.info(
        f'device: {device}, pooling: {cfg.REPRESENTATION.SIMCSE.POOLING}, model path: {model_path}'
    )
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # load data
    train_data = load_data(cfg.REPRESENTATION.SIMCSE.TRAIN_DATA)
    random.shuffle(train_data)
    dev_data = load_data(cfg.REPRESENTATION.SIMCSE.EVAL_DATA)
    test_data = load_data(cfg.REPRESENTATION.SIMCSE.TEST_DATA)
    train_dataloader = DataLoader(TrainDataset(train_data),
                                  batch_size=batch_size)
    dev_dataloader = DataLoader(TestDataset(dev_data), batch_size=batch_size)
    test_dataloader = DataLoader(TestDataset(test_data), batch_size=batch_size)
    # load model
    assert cfg.REPRESENTATION.SIMCSE.POOLING in [
        'cls', 'pooler', 'last-avg', 'first-last-avg'
    ]
    model = SimcseModel(cfg)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.REPRESENTATION.SIMCSE.LR)
    # train
    best = 0
    for epoch in range(cfg.REPRESENTATION.SIMCSE.EPOCHS):
        logger.info(f'epoch: {epoch}')
        train(model, train_dataloader, dev_dataloader, optimizer, device,
              save_path, batch_size)
    logger.info(f'train is finished, best model is saved at {save_path}')
    # eval
    model.load_state_dict(torch.load(save_path))
    dev_corrcoef = eval(model, dev_dataloader, device)
    test_corrcoef = eval(model, test_dataloader, device)
    logger.info(f'dev_corrcoef: {dev_corrcoef:.4f}')
    logger.info(f'test_corrcoef: {test_corrcoef:.4f}')
