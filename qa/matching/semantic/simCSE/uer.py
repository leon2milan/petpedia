import argparse
import json
import math
import os
import random
import sys
from argparse import Namespace

import numpy as np
import torch
from torch.autograd.grad_mode import no_grad
import torch.nn as nn
from qa.matching.semantic.simCSE.embedding import (DualEmbedding,
                                                   WordEmbedding,
                                                   WordPosEmbedding,
                                                   WordPosSegEmbedding,
                                                   WordSinusoidalposEmbedding)
from qa.matching.semantic.simCSE.optimizers import (
    Adafactor, AdamW, get_constant_schedule, get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup)
from qa.matching.semantic.simCSE.opts import finetune_opts, tokenizer_opts
from qa.matching.semantic.simCSE.tokenizer import (
    CLS_TOKEN, MASK_TOKEN, PAD_TOKEN, SEP_TOKEN, UNK_TOKEN, BertTokenizer,
    CharTokenizer, SpaceTokenizer, XLMRobertaTokenizer)
from qa.matching.semantic.simCSE.transformer_encoder import TransformerEncoder

str2optimizer = {"adamw": AdamW, "adafactor": Adafactor}

str2scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_with_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule,
    "constant_with_warmup": get_constant_schedule_with_warmup
}
str2encoder = {"transformer": TransformerEncoder}
str2embedding = {
    "word": WordEmbedding,
    "word_pos": WordPosEmbedding,
    "word_pos_seg": WordPosSegEmbedding,
    "word_sinusoidalpos": WordSinusoidalposEmbedding,
    "dual": DualEmbedding
}
str2tokenizer = {
    "char": CharTokenizer,
    "space": SpaceTokenizer,
    "bert": BertTokenizer,
    "xlmroberta": XLMRobertaTokenizer
}


def set_seed(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def save_model(model, model_path):
    """
    Save model weights to file.
    """
    if hasattr(model, "module"):
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)


def load_hyperparam(default_args):
    """
    Load arguments form argparse and config file
    Priority: default options < config file < command line args
    """
    with open(default_args.config_path, mode="r", encoding="utf-8") as f:
        config_args_dict = json.load(f)

    default_args_dict = vars(default_args)

    command_line_args_dict = {
        k: default_args_dict[k]
        for k in
        [a[2:] for a in sys.argv if (a[:2] == "--" and "local_rank" not in a)]
    }
    default_args_dict.update(config_args_dict)
    default_args_dict.update(command_line_args_dict)
    args = Namespace(**default_args_dict)

    return args


def batch_loader(batch_size, src, tgt, seg):
    instances_num = tgt.size()[0]
    src_a, src_b = src
    seg_a, seg_b = seg
    for i in range(instances_num // batch_size):
        src_a_batch = src_a[i * batch_size:(i + 1) * batch_size, :]
        src_b_batch = src_b[i * batch_size:(i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size:(i + 1) * batch_size]
        seg_a_batch = seg_a[i * batch_size:(i + 1) * batch_size, :]
        seg_b_batch = seg_b[i * batch_size:(i + 1) * batch_size, :]
        yield (src_a_batch, src_b_batch), tgt_batch, (seg_a_batch, seg_b_batch)
    if instances_num > instances_num // batch_size * batch_size:
        src_a_batch = src_a[instances_num // batch_size * batch_size:, :]
        src_b_batch = src_b[instances_num // batch_size * batch_size:, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size:]
        seg_a_batch = seg_a[instances_num // batch_size * batch_size:, :]
        seg_b_batch = seg_b[instances_num // batch_size * batch_size:, :]
        yield (src_a_batch, src_b_batch), tgt_batch, (seg_a_batch, seg_b_batch)


def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [{
        'params':
        [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':
        0.01
    }, {
        'params':
        [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
        0.0
    }]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters,
                                                  lr=args.learning_rate,
                                                  correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters,
                                                  lr=args.learning_rate,
                                                  scale_parameter=False,
                                                  relative_step=False)
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps *
                                                  args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](
            optimizer, args.train_steps * args.warmup, args.train_steps)
    return optimizer,


class SimCSE(nn.Module):
    def __init__(self, args):
        super(SimCSE, self).__init__()
        self.embedding = str2embedding[args.embedding](
            args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)

        self.pooling_type = args.pooling

    def forward(self, src, seg):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        # Embedding.
        emb_0 = self.embedding(src[0], seg[0])
        emb_1 = self.embedding(src[1], seg[1])
        # Encoder.
        output_0 = self.encoder(emb_0, seg[0])
        output_1 = self.encoder(emb_1, seg[1])
        # Target.
        features_0 = self.pooling(output_0, seg[0], self.pooling_type)
        features_1 = self.pooling(output_1, seg[1], self.pooling_type)

        return features_0, features_1

    def pooling(self, memory_bank, seg, pooling_type):
        seg = torch.unsqueeze(seg, dim=-1).type(torch.float)
        memory_bank = memory_bank * seg
        if pooling_type == "mean":
            features = torch.sum(memory_bank, dim=1)
            features = torch.div(features, torch.sum(seg, dim=1))
        elif pooling_type == "last":
            features = memory_bank[
                torch.arange(memory_bank.shape[0]),
                torch.squeeze(torch.sum(seg, dim=1).type(torch.int64) - 1), :]
        elif pooling_type == "max":
            features = torch.max(memory_bank + (seg - 1) * sys.maxsize,
                                 dim=1)[0]
        else:
            features = memory_bank[:, 0, :]
        return features


def similarity(x, y, temperature):
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    return torch.matmul(x, y.transpose(-2, -1)) / temperature


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    tokenizer_opts(parser)

    parser.add_argument("--pooling",
                        choices=["mean", "max", "first", "last"],
                        default="first",
                        help="Pooling type.")
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--eval_steps",
                        type=int,
                        default=200,
                        help="Evaluate frequency.")

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)
    return args


class UER:
    def __init__(self) -> None:
        self.args = get_args()
        set_seed(self.args.seed)
        self.args.tokenizer = str2tokenizer[self.args.tokenizer](self.args)
        self.model = SimCSE(self.args)
        self.load_or_initialize_parameters()

        self.args.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.args.device)
        self.batch_size = self.args.batch_size
        self.model.eval()

        if torch.cuda.device_count() > 1:
            print("{} GPUs are available. Let's use them.".format(
                torch.cuda.device_count()))
            model = torch.nn.DataParallel(self.model)
        self.args.model = model

    def load_or_initialize_parameters(self):
        if self.args.pretrained_model_path is not None:
            # Initialize with pretrained model.
            self.model.load_state_dict(torch.load(
                self.args.pretrained_model_path),
                                       strict=False)
        else:
            # Initialize with normal distribution.
            for n, p in list(self.model.named_parameters()):
                if "gamma" not in n and "beta" not in n:
                    p.data.normal_(0, 0.02)

    def trans_data(self, text_a, text_b):
        src_a = self.args.tokenizer.convert_tokens_to_ids(
            [CLS_TOKEN] + self.args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
        src_b = self.args.tokenizer.convert_tokens_to_ids(
            [CLS_TOKEN] + self.args.tokenizer.tokenize(text_b) + [SEP_TOKEN])
        seg_a = [1] * len(src_a)
        seg_b = [1] * len(src_b)
        PAD_ID = self.args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]

        if len(src_a) >= self.args.seq_length:
            src_a = src_a[:self.args.seq_length]
            seg_a = seg_a[:self.args.seq_length]
        while len(src_a) < self.args.seq_length:
            src_a.append(PAD_ID)
            seg_a.append(0)

        if len(src_b) >= self.args.seq_length:
            src_b = src_b[:self.args.seq_length]
            seg_b = seg_b[:self.args.seq_length]
        while len(src_b) < self.args.seq_length:
            src_b.append(PAD_ID)
            seg_b.append(0)

        data = ((src_a, src_a), -1, (seg_a, seg_a))
        return data

    def evaluate(self, dataset):
        src_a = torch.LongTensor([example[0][0] for example in dataset])
        src_b = torch.LongTensor([example[0][1] for example in dataset])
        tgt = torch.FloatTensor([example[1] for example in dataset])
        seg_a = torch.LongTensor([example[2][0] for example in dataset])
        seg_b = torch.LongTensor([example[2][1] for example in dataset])

        all_similarities = []
        batch_size = self.args.batch_size
        with no_grad():
            for i, (src_batch, _, seg_batch) in enumerate(
                    batch_loader(batch_size, (src_a, src_b), tgt, (seg_a, seg_b))):

                src_a_batch, src_b_batch = src_batch
                seg_a_batch, seg_b_batch = seg_batch

                src_a_batch = src_a_batch.to(self.args.device)
                src_b_batch = src_b_batch.to(self.args.device)

                seg_a_batch = seg_a_batch.to(self.args.device)
                seg_b_batch = seg_b_batch.to(self.args.device)

                with torch.no_grad():
                    features_0, features_1 = self.args.model(
                        (src_a_batch, src_b_batch), (seg_a_batch, seg_b_batch))
                similarity_matrix = similarity(features_0, features_1, 1)

                for j in range(similarity_matrix.size(0)):
                    all_similarities.append(similarity_matrix[j][j].item())

        return all_similarities

    def similarity(self, str1, str2):
        data = self.trans_data(str1, str2)
        return self.evaluate([data])


if __name__ == "__main__":
    uer = UER()
    import time
    test = [['哈士奇拆家怎么办', '猫拉肚子不吃东西'], ['哈士奇拆家怎么办', '哈士奇总是乱咬东西怎么办'],
            ['哈士奇拆家怎么办', '狗拆家怎么办'], ['猫癣', '什么是猫癣'], ['什么是猫癣', '得了猫癣怎么办'],
            ['猫癣', '猫癣转阴'], ['聪明', '智商'], ['我家猫半夜瞎叫唤，咋办？', '猫半夜叫唤咋回事'],
            ['我家猫半夜瞎叫唤，咋办？', '猫咪发情咋办'], ['我家猫拉稀了， 怎么办', '猫尿道炎拉稀吗'],
            ['我家猫拉稀了， 怎么办', '猫为什么拉稀'], ['我想养个狗，应该注意什么？', '养狗狗应该注意什么'],
            ['我想养个狗，应该注意什么？', '养杜宾要注意什么'], ['我想养个哈士奇，应该注意什么？', '养狗狗应该注意什么'],
            ['我想养个哈士奇，应该注意什么？', '养哈士奇有什么必须留意的吗？你知道吗']]

    for i in test:
        s = time.time()
        print(
            f"query: {i[0]}, candidate: {i[1]}, score: {uer.similarity(i[0], i[1])}, time: {time.time() - s}"
        )
