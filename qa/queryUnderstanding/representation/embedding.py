from abc import ABCMeta, abstractmethod
from config import get_cfg
from qa.queryUnderstanding.querySegmentation import Segmentation
from lmdb_embeddings.exceptions import MissingWordError
import numpy as np

__all__ = ['Embedding']


class Embedding(metaclass=ABCMeta):
    __slot__ = ['cfg', 'seg']

    def __init__(self, cfg):
        self.cfg = cfg
        self.seg = Segmentation(self.cfg)

    @staticmethod
    def get_embedding(w2v_model, x, USE_LMDB=True, embedding_size=300):
        if USE_LMDB:
            try:
                return w2v_model.get_word_vector(x)
            except MissingWordError:
                # 'google' is not in the database.
                return np.zeros((embedding_size, ))
        else:
            if x in w2v_model.wv.vocab.keys():
                return w2v_model.wv.get_vector(x)
            else:
                return np.zeros((embedding_size, ))

    @staticmethod
    def wam(sentences,
            w2v_model,
            USE_LMDB=True,
            agg='mean',
            embedding_size=300):
        '''
        @description: 通过word average model 生成句向量
        @param {type}
        sentence: 以空格分割的句子
        w2v_model: word2vec模型
        @return:
        '''
        if not any(isinstance(el, list) for el in sentences):
            sentences = [sentences]
        func = np.mean if agg == 'mean' else np.sum
        arr = np.stack([
            func([
                Embedding.get_embedding(w2v_model, s, USE_LMDB, embedding_size)
                for s in sentence
            ],
                 axis=0) for sentence in sentences
        ])
        return arr

    @abstractmethod
    def get_embedding_helper(self):
        pass