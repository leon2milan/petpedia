# -*- encoding: utf-8 -*-

import random
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer
from config.config import get_cfg

from qa.tools import setup_logger

logger = setup_logger()


def load_data(name: str, path: str) -> List:
    def load_self(path):
        tmp = []
        with open(path, 'r') as f:
            for line in f.readlines():
                tmp.append("".join(line.strip().split()))
        return tmp

    def load_sts_data(path):
        with open(path, 'r', encoding='utf8') as f:
            return [(line.split("||")[1], line.split("||")[2],
                     line.split("||")[3]) for line in f]

    if name == 'self':
        return load_self(path)
    return load_sts_data(path)


class TrainDataset(Dataset):
    """训练数据集, 重写__getitem__和__len__方法"""
    def __init__(self, cfg, data: List):
        self.cfg = cfg
        self.data = data

    def __len__(self):
        return len(self.data)

    def text_2_id(self, text: str):
        # 添加自身两次, 经过bert编码之后, 互为正样本
        return tokenizer([text, text],
                         max_length=self.cfg.REPRESENTATION.SIMCSE.MAXLEN,
                         truncation=True,
                         padding='max_length',
                         return_tensors='pt')

    def __getitem__(self, index: int):
        return self.text_2_id(self.data[index])


class TestDataset(Dataset):
    """测试数据集, 重写__getitem__和__len__方法"""
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

    def __getitem__(self, index: int):
        da = self.data[index]
        return self.text_2_id([da[0]]), self.text_2_id([da[1]]), int(da[2])


class SimcseModel(nn.Module):
    """Simcse无监督模型定义"""
    def __init__(self, cfg):
        super(SimcseModel, self).__init__()
        self.cfg = cfg
        config = BertConfig.from_pretrained(
            self.cfg.REPRESENTATION.SIMCSE.PRETRAINED_MODEL)
        config.attention_probs_dropout_prob = self.cfg.REPRESENTATION.SIMCSE.DROPOUT  # 修改config的dropout系数
        config.hidden_dropout_prob = self.cfg.REPRESENTATION.SIMCSE.DROPOUT
        self.bert = BertModel.from_pretrained(
            self.cfg.REPRESENTATION.SIMCSE.PRETRAINED_MODEL, config=config)
        self.pooling = self.cfg.REPRESENTATION.SIMCSE.POOLING

    def forward(self, input_ids, attention_mask, token_type_ids):

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


def simcse_unsup_loss(y_pred: 'tensor', device) -> 'tensor':
    """无监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 2, 768]
    
    """
    # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
    y_true = torch.arange(y_pred.shape[0], device=device)
    y_true = (y_true - y_true % 2 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
    # 相似度矩阵除以温度系数
    sim = sim / 0.05
    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(sim, y_true)
    return torch.mean(loss)


def eval(model, dataloader, device) -> float:
    """模型评估函数 
    批量预测, batch结果拼接, 一次性求spearman相关度
    """
    model.eval()
    sim_tensor = torch.tensor([], device=device)
    label_array = np.array([])
    with torch.no_grad():
        for source, target, label in dataloader:
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source.get('input_ids').squeeze(1).to(device)
            source_attention_mask = source.get('attention_mask').squeeze(1).to(
                device)
            source_token_type_ids = source.get('token_type_ids').squeeze(1).to(
                device)
            source_pred = model(source_input_ids, source_attention_mask,
                                source_token_type_ids)
            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target.get('input_ids').squeeze(1).to(device)
            target_attention_mask = target.get('attention_mask').squeeze(1).to(
                device)
            target_token_type_ids = target.get('token_type_ids').squeeze(1).to(
                device)
            target_pred = model(target_input_ids, target_attention_mask,
                                target_token_type_ids)
            # concat
            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            label_array = np.append(label_array, np.array(label))
    # corrcoef
    return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation


def train(model, train_dl, dev_dl, optimizer, device, save_path) -> None:
    """模型训练函数"""
    model.train()
    global best
    for batch_idx, source in enumerate(tqdm(train_dl), start=1):
        # 维度转换 [batch, 2, seq_len] -> [batch * 2, sql_len]
        real_batch_num = source.get('input_ids').shape[0]
        input_ids = source.get('input_ids').view(real_batch_num * 2,
                                                 -1).to(device)
        attention_mask = source.get('attention_mask').view(
            real_batch_num * 2, -1).to(device)
        token_type_ids = source.get('token_type_ids').view(
            real_batch_num * 2, -1).to(device)

        out = model(input_ids, attention_mask, token_type_ids)
        loss = simcse_unsup_loss(out, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 1000 == 0:
            corrcoef = eval(model, dev_dl, device)
            logger.info(f'loss: {loss.item():.4f}， corrcoef: {corrcoef:.4f}')
            torch.save(model.state_dict(), save_path)
            model.train()
            if best < corrcoef:
                best = corrcoef
                logger.info(
                    f"higher corrcoef: {best:.4f} in batch: {batch_idx}, save model"
                )


if __name__ == '__main__':
    cfg = get_cfg()
    device = torch.device(
        'cuda' if cfg.REPRESENTATION.SIMCSE.DEVICE else 'cpu')
    model_path = cfg.REPRESENTATION.SIMCSE.PRETRAINED_MODEL
    batch_size = cfg.REPRESENTATION.SIMCSE.BATCH_SIZE
    save_path = cfg.REPRESENTATION.SIMCSE.SAVE_PATH
    logger.info(
        f'device: {device}, pooling: {cfg.REPRESENTATION.SIMCSE.POOLING}, model path: {model_path}'
    )
    tokenizer = BertTokenizer.from_pretrained(model_path)
    # load data
    train_data = load_data('self', cfg.REPRESENTATION.SIMCSE.TRAIN_DATA)
    random.shuffle(train_data)

    dev_data = load_data('sts', cfg.REPRESENTATION.SIMCSE.EVAL_DATA)
    train_dataloader = DataLoader(TrainDataset(cfg, train_data),
                                  batch_size=batch_size)
    dev_dataloader = DataLoader(TestDataset(cfg, dev_data),
                                batch_size=batch_size)
    # load model
    assert cfg.REPRESENTATION.SIMCSE.POOLING in [
        'cls', 'pooler', 'last-avg', 'first-last-avg'
    ]
    model = SimcseModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.REPRESENTATION.SIMCSE.LR)
    # train
    best = 0
    for epoch in range(cfg.REPRESENTATION.SIMCSE.EPOCHS):
        logger.info(f'epoch: {epoch}')
        train(model, train_dataloader, dev_dataloader, optimizer, device,
              save_path)
    logger.info(f'train is finished, best model is saved at {save_path}')
    # eval
    model.load_state_dict(torch.load(save_path))
    dev_corrcoef = eval(model, dev_dataloader, device)
    logger.info(f'dev_corrcoef: {dev_corrcoef:.4f}')