from collections import deque

import numpy as np
from config.config import get_cfg

np.random.seed(12345)
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from qa.queryUnderstanding.queryReformat.queryCorrection.pinyin import Pinyin
from qa.queryUnderstanding.querySegmentation import Words
from qa.tools.wubi import wubi
from torch.autograd import Variable
from tqdm import tqdm


def pad(data, pad=0):
    max_len = max([len(x) for x in data])
    return [x + [pad] * (max_len - len(x)) for x in data]


class InputData:
    """Store data for word2vec, such as word map, sampling table and so on.
    Attributes:
        word_frequency: Count of each word, used for filtering low-frequency words and sampling table
        word2id: Map from word to word id, without low-frequency words.
        id2word: Map from word id to word, without low-frequency words.
        sentence_count: Sentence count in files.
        word_count: Word count in files, without low-frequency words.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.same_pinyin = Words(cfg).get_samepinyin
        self.same_stroke = Words(cfg).get_samestroke
        self.input_file_name = self.cfg.BASE.CHAR_FILE
        try:
            self.load()
        except:
            self.get_words(self.cfg.CORRECTION.THRESHOLD)
        self.word_pair_catch = deque()
        print('Word Count: %d' % len(self.word2id))
        print('Sentence Length: %d' % (self.sentence_length))

    def get_words(self, min_count):
        self.input_file = open(self.input_file_name)
        self.sentence_length = 0
        self.sentence_count = 0
        word_frequency = dict()
        for line in self.input_file:
            self.sentence_count += 1
            line = line.strip().split(' ')
            self.sentence_length += len(line)
            for w in line:
                try:
                    word_frequency[w] += 1
                except:
                    word_frequency[w] = 1
        self.word2id = {self.cfg.BASE.PAD_TOKEN: 0}
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.py2id = dict()
        self.id2py = dict()
        self.stroke2id = dict()
        self.id2stroke = dict()
        wid = 0
        pid = 0
        sid = 0
        self.word_frequency = dict()
        for w, c in word_frequency.items():
            if c < min_count:
                self.sentence_length -= c
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            py = "".join(Pinyin.get_pinyin_list(w))
            if py not in self.py2id:
                self.py2id[py] = pid
                self.id2py[pid] = py
                pid += 1
            stroke = wubi.get(w, 'cw')
            if stroke not in self.stroke2id:
                self.stroke2id[stroke] = sid
                self.id2stroke[sid] = stroke
                sid += 1
            wid += 1

        self.word_count = len(self.word2id)
        self.save()

    def save(self):
        with open(
                os.path.join(self.cfg.BASE.MODEL_PATH,
                             'embedding/word2id.json'), 'w') as f:
            json.dump(self.word2id, f, ensure_ascii=False)
        with open(
                os.path.join(self.cfg.BASE.MODEL_PATH, 'embedding/py2id.json'),
                'w') as f:
            json.dump(self.py2id, f, ensure_ascii=False)
        with open(
                os.path.join(self.cfg.BASE.MODEL_PATH,
                             'embedding/stroke2id.json'), 'w') as f:
            json.dump(self.stroke2id, f, ensure_ascii=False)
        with open(
                os.path.join(self.cfg.BASE.MODEL_PATH,
                             'embedding/charfreq.json'), 'w') as f:
            json.dump(self.word_frequency, f, ensure_ascii=False)

    def load(self):
        with open(os.path.join(self.cfg.BASE.MODEL_PATH, 'word2id.json')) as f:
            self.word2id = json.load(f)
            self.id2word = {v: k for k, v in self.word2id.items()}
        with open(os.path.join(self.cfg.BASE.MODEL_PATH, 'py2id.json')) as f:
            self.py2id = json.load(f)
            self.id2py = {v: k for k, v in self.py2id.items()}
        with open(os.path.join(self.cfg.BASE.MODEL_PATH,
                               'stroke2id.json')) as f:
            self.stroke2id = json.load(f)
            self.id2stroke = {v: k for k, v in self.stroke2id.items()}
        with open(os.path.join(self.cfg.BASE.MODEL_PATH,
                               'charfreq.json')) as f:
            self.word_frequency = json.load(f)

    # @profile
    def get_cbow_batch_all_pairs(self, window_size):
        sentences = open(self.input_file_name).readlines()
        for sentence in sentences:
            if sentence is None or sentence == '':
                continue
            word_ids = []
            for word in sentence.strip().split(' '):
                try:
                    word_ids.append(self.word2id[word])
                except:
                    continue
            for i, u in enumerate(word_ids):
                contentw = []
                for j, v in enumerate(word_ids):
                    assert u < self.word_count
                    assert v < self.word_count
                    if i == j:
                        continue
                    elif j >= max(0, i - window_size + 1) and j <= min(
                            len(word_ids), i + window_size - 1):
                        contentw.append(v)
                if len(contentw) == 0:
                    continue
                py = self.py2id["".join(Pinyin.get_pinyin_list(
                    self.id2word[u]))]
                stroke = self.stroke2id[wubi.get(self.id2word[u], 'cw')]
                self.word_pair_catch.append((contentw, py, stroke, u))
        return self.word_pair_catch

    def get_cbow_batch_pairs(self, batch_size):
        batch_pairs = []
        for _ in range(batch_size):
            batch_pairs.append(self.word_pair_catch.popleft())
        return batch_pairs


class CustomW2vModel(nn.Module):
    def __init__(self, cfg, vocab_size, py_size, stroke_size):

        super(CustomW2vModel, self).__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.emb_dimension = self.cfg.REPRESENTATION.CUSTOM_W2V.EMB_SIZE
        self.word_embeddings = nn.Embedding(vocab_size, self.emb_dimension)
        self.py_embeddings = nn.Embedding(py_size, self.emb_dimension)
        self.stroke_embeddings = nn.Embedding(stroke_size, self.emb_dimension)
        self.fc1 = nn.Linear(self.emb_dimension * 3, self.emb_dimension * 3)
        self.fc2 = nn.Linear(self.emb_dimension * 3, self.emb_dimension * 3)
        self.fc3 = nn.Linear(self.emb_dimension * 3, self.vocab_size)
        self.init_emb()

    def init_emb(self):
        """Initialize embedding weight like word2vec.
        The u_embedding is a uniform distribution in [-0.5/em_size, 0.5/emb_size], and the elements of v_embedding are zeroes.
        Returns:
            None
        """
        initrange = 0.5 / self.emb_dimension
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)
        self.py_embeddings.weight.data.uniform_(-initrange, initrange)
        self.stroke_embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, content, pinyin, stroke):
        word_emb = []
        for i in range(content.shape[0]):
            word_emb.append(
                torch.sum(self.word_embeddings(content[i]),
                          axis=0,
                          keepdim=True))

        word_emb = torch.cat(word_emb)
        pinyin_emb = self.py_embeddings(pinyin)
        stroke_emb = self.stroke_embeddings(stroke)
        score = torch.cat([word_emb, pinyin_emb, stroke_emb], 1)
        score = F.relu(self.fc1(score))
        score = F.relu(self.fc2(score))
        return self.fc3(score)


class Word2Vec:
    def __init__(self,
                 cfg,
                 batch_size=64,
                 window_size=5,
                 iteration=10,
                 initial_lr=0.001):
        """Initilize class parameters.
        Args:
            input_file_name: Name of a text data from file. Each line is a sentence splited with space.
            output_file_name: Name of the final embedding file.
            emb_dimention: Embedding dimention, typically from 50 to 500.
            batch_size: The count of word pairs for one forward.
            window_size: Max skip length between words.
            iteration: Control the multiple training iterations.
            initial_lr: Initial learning rate.
            min_count: The minimal word frequency, words with lower frequency will be filtered.
        Returns:
            None.
        """
        self.cfg = cfg
        self.data = InputData(self.cfg)
        self.output_file_name = self.cfg.REPRESENTATION.CUSTOM_W2V.SAVE_PATH
        self.word_size = len(self.data.word2id)
        self.py_size = len(self.data.py2id)
        self.stroke_size = len(self.data.stroke2id)
        self.emb_dimension = self.cfg.REPRESENTATION.CUSTOM_W2V.EMB_SIZE
        self.batch_size = batch_size
        self.window_size = window_size
        self.iteration = iteration
        self.initial_lr = initial_lr
        self.model = CustomW2vModel(self.cfg, self.word_size, self.py_size,
                                    self.stroke_size)
        if os.path.exists(self.output_file_name):
            self.model.load_state_dict(torch.load(self.output_file_name))
            self.model.eval()
            print("Load model successful!!!")
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.model.cuda()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.initial_lr)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        """Multiple training.
        Returns:
            None.
        """
        pos_all_pairs = self.data.get_cbow_batch_all_pairs(self.window_size)
        pair_count = len(pos_all_pairs)
        batch_count = self.iteration * pair_count / self.batch_size
        process_bar = tqdm(range(int(batch_count)))
        # self.skip_gram_model.save_embedding(
        #     self.data.id2word, 'begin_embedding.txt', self.use_cuda)
        for i in process_bar:
            pos_pairs = self.data.get_cbow_batch_pairs(self.batch_size)
            content = Variable(
                torch.LongTensor(pad([pair[0] for pair in pos_pairs])))
            pinyin = Variable(torch.LongTensor([pair[1]
                                                for pair in pos_pairs]))
            stroke = Variable(torch.LongTensor([pair[2]
                                                for pair in pos_pairs]))
            golden = Variable(torch.LongTensor([pair[3]
                                                for pair in pos_pairs]))

            if self.use_cuda:
                content = content.cuda()
                pinyin = pinyin.cuda()
                stroke = stroke.cuda()
                golden = golden.cuda()

            self.optimizer.zero_grad()
            pred = self.model.forward(content, pinyin, stroke)
            loss = self.criterion(pred, golden)
            loss.backward()
            self.optimizer.step()

            process_bar.set_description(
                "Loss: %0.8f, lr: %0.6f" %
                (loss.item(), self.optimizer.param_groups[0]['lr']))
            if i * self.batch_size % 100000 == 0:
                lr = self.initial_lr * (1.0 - 1.0 * i / batch_count)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
        torch.save(self.model.state_dict(), self.output_file_name)

    def predict(self, text):
        word_ids = []
        for word in list(text.strip()):
            try:
                word_ids.append(self.data.word2id[word])
            except:
                continue
        inputs = []
        for i, u in enumerate(word_ids):
            contentw = []
            for j, v in enumerate(word_ids):
                if i == j:
                    continue
                elif j >= max(0, i - self.window_size + 1) and j <= min(
                        len(word_ids), i + self.window_size - 1):
                    contentw.append(v)
            py = self.data.py2id["".join(
                Pinyin.get_pinyin_list(self.data.id2word[u]))]
            stroke = self.data.stroke2id[wubi.get(self.data.id2word[u], 'cw')]

            inputs.append((contentw, py, stroke, u))
        print('inputs', inputs)
        content = Variable(torch.LongTensor(pad([pair[0] for pair in inputs])))
        pinyin = Variable(torch.LongTensor([pair[1] for pair in inputs]))
        stroke = Variable(torch.LongTensor([pair[2] for pair in inputs]))
        golden = Variable(torch.LongTensor([pair[3] for pair in inputs]))
        if self.use_cuda:
            content = content.cuda()
            pinyin = pinyin.cuda()
            stroke = stroke.cuda()
            golden = golden.cuda()
        pred = self.model.forward(content, pinyin, stroke)
        for i in range(pred.shape[0]):
            print('golden', golden[i].cpu().item(), self.data.id2word[golden[i].cpu().item()])
            print('golden score', pred[i][golden[i].cpu().item()].cpu().item())
            print('pred score', pred[i].argmax(), pred[i][pred[i].argmax()].cpu().item(), 
                  self.data.id2word[pred[i].argmax().cpu().item()])
            print('tok 10', pred[i].topk(10)[1].cpu())


if __name__ == "__main__":
    cfg = get_cfg()
    w2v = Word2Vec(cfg)
    # w2v.train()
    text = [
        "猫沙盆",  # 错字
        "我家猫猫精神没有",  # 乱序
        "狗狗发了",  # 少字
        "maomi",  # 拼音
        "猫咪啦肚子怎么办",
        "我家狗拉啦稀了",  # 多字
        "狗狗偶尔尿xie怎么办",
        "狗老抽出怎么办",
        '我家猫猫感帽了',
        '传然给我',
        '呕土不止',
        '一直咳数',
        '我想找哥医生'
    ]
    for i in text:
        w2v.predict(i)
