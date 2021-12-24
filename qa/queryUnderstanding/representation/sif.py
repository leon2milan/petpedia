from joblib import dump, load
import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.decomposition import IncrementalPCA
from config import get_cfg
from qa.queryUnderstanding.representation import W2V, Embedding, REPRESENTATION_REGISTRY
from qa.tools.logger import setup_logger
from qa.tools.utils import Singleton

logging = setup_logger()
__all__ = ['SIF']


@REPRESENTATION_REGISTRY.register()
@Singleton
class SIF(Embedding):
    __slot__ = [
        'cfg', 'a', 'rmpc', 'is_rough', 'model_path', 'data_path',
        'weightfile', 'w2v', 'idx2weight', 'vocab_size', 'embedding_size', 'U'
    ]

    def __init__(self, cfg, is_rough=False):
        Embedding.__init__(self, cfg)
        self.a = self.cfg.REPRESENTATION.SIF.A
        self.rmpc = self.cfg.REPRESENTATION.SIF.RMPC
        self.is_rough = is_rough
        if self.is_rough:
            self.model_path = self.cfg.REPRESENTATION.SIF.ROUGH_PCA_PATH
            self.data_path = self.cfg.BASE.ROUGH_WORD_FILE
            self.weightfile = self.cfg.REPRESENTATION.SIF.ROUGH_WEIGHTFILE_PATH
        else:
            self.model_path = self.cfg.REPRESENTATION.SIF.FINE_PCA_PATH
            self.data_path = self.cfg.BASE.FINE_WORD_FILE
            self.weightfile = self.cfg.REPRESENTATION.SIF.FINE_WEIGHTFILE_PATH
        if not os.path.exists(self.weightfile):
            logging.info('need count word freq')
            self.getWeightFile(self.data_path, self.weightfile)
        self.w2v = W2V(self.cfg, is_rough=self.is_rough, gensim_way=False)
        self.idx2weight = self.getWeight()
        self.vocab_size = len(self.w2v.word2idx)
        self.embedding_size = self.w2v.idx2embedding.shape[1]

        if not os.path.exists(self.model_path):
            self.U = self.train_pca()
        else:
            self.U = self.load_pca()
            if len(self.U) < self.embedding_size:
                for i in range(self.embedding_size - len(self.U)):
                    self.U = np.append(
                        self.U,
                        0)  # add needed extension for multiplication below

    def train_pca(self):
        logging.info("load training data")
        chunksize_ = 5 * 25000
        reader = pd.read_table(self.data_path,
                               header=None,
                               names=['content'],
                               chunksize=chunksize_,
                               iterator=True)
        pca = IncrementalPCA()
        # sentences = [x.split() for x in open(self.data_path).read()]
        logging.info('incremental train  PCA model')
        from tqdm import tqdm
        for chunk in tqdm(reader):
            sentences = chunk.pop('content')
            seqs, weights = self.seq2weight(sentences)
            embedding = self.getEmb(seqs, weights, train=True)
            embedding = embedding[~np.any(np.isnan(embedding), axis=1)]
            pca.partial_fit(np.array(embedding))

        u = pca.components_[0]  # the PCA vector
        u = np.multiply(u, np.transpose(u))  # u x uT
        logging.info("save pca model")
        dump(pca, self.model_path)
        return u

    def load_pca(self):
        logging.info("load pca model")
        pca = load(self.model_path)
        u = pca.components_[0]  # the PCA vector
        u = np.multiply(u, np.transpose(u))  # u x uT
        return u

    @staticmethod
    def getWeightFile(data_path, weightfile):
        sentences = [
            y for x in open(data_path).readlines() for y in x.strip().split()
        ]

        freq = Counter(sentences)
        freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        with open(weightfile, 'w') as f:
            for k, v in freq:
                if k and v:
                    f.write(k + '\t' + str(v) + '\n')

    def getWordWeight(self):
        logging.info("format word weight")
        word2weight = {}
        with open(self.weightfile) as f:
            lines = f.readlines()
        N = 0
        for i in lines:
            i = i.strip()
            if (len(i) > 0):
                i = i.split()
                if (len(i) == 2):
                    word2weight[i[0]] = float(i[1])
                    N += float(i[1])
                else:
                    logging.info(i)
        for key, value in word2weight.items():
            word2weight[key] = self.a / (self.a + value / N)
        return word2weight

    def getWeight(self):
        word2weight = self.getWordWeight()

        idx2weight = {}
        for word, index in self.w2v.word2idx.items():
            if word in word2weight:
                idx2weight[index] = word2weight[word]
            # else:
            #     idx2weight[index] = 1.0
        return idx2weight

    def getEmb(self, seq, mask, train=False):
        n_samples = seq.shape[0]
        emb = np.zeros((n_samples, self.embedding_size))
        for i in range(n_samples):
            zero = [
                True if seq[i, x] in self.idx2weight else False
                for x in range(len(seq[i, :]))
            ]
            embedding = self.w2v.idx2embedding[seq[i, :]][zero]
            m = mask[i, :][zero]
            emb[i, :] = m.dot(embedding) / np.count_nonzero(m)
        if self.rmpc > 0 and not train:
            emb = emb - np.multiply(emb, self.U)
        return emb

    def get_embedding_helper(self, s):
        s = [s] if isinstance(s, str) else s
        s, m = self.seq2weight(s)
        return self.getEmb(s, m)

    def similarity(self, s1, s2):
        emb1 = self.get_embedding_helper(s1)
        emb2 = self.get_embedding_helper(s2)

        inn = (emb1 * emb2).sum(axis=1)
        emb1norm = np.sqrt((emb1 * emb1).sum(axis=1))
        emb2norm = np.sqrt((emb2 * emb2).sum(axis=1))
        scores = inn / emb1norm / emb2norm
        return scores

    def getSeqs(self, sentence):
        res = []
        sentence = sentence.split() if isinstance(sentence, str) else sentence
        for i in sentence:
            if i in self.w2v.word2idx:
                res.append(self.w2v.word2idx[i])
            # else:
            #     res.append(self.vocab_size + 1)
        return res

    @staticmethod
    def prepare_data(list_of_seqs):
        lengths = [len(s) for s in list_of_seqs]
        n_samples = len(list_of_seqs)
        maxlen = np.max(lengths)
        x = np.zeros((n_samples, maxlen)).astype('int32')
        x_mask = np.zeros((n_samples, maxlen)).astype('float32')
        for idx, seq in enumerate(list_of_seqs):
            x[idx, :lengths[idx]] = seq
            x_mask[idx, :lengths[idx]] = 1.
        x_mask = np.asarray(x_mask, dtype='float32')
        return x, x_mask

    def seq2weight(self, sentences):
        seq = []
        for s in sentences:
            X1 = self.getSeqs(s)
            seq.append(X1)

        seq, mask = SIF.prepare_data.__func__(seq)
        weight = np.zeros(seq.shape).astype('float32')
        for i in range(seq.shape[0]):
            for j in range(seq.shape[1]):
                if mask[i, j] > 0 and seq[i, j] in self.idx2weight:
                    weight[i, j] = self.idx2weight[seq[i, j]]
        weight = np.asarray(weight, dtype='float32')
        return seq, weight


if __name__ == '__main__':
    cfg = get_cfg()
    SIF(cfg, is_rough=True)
    SIF(cfg, is_rough=False)
