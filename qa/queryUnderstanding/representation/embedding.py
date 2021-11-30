from abc import ABCMeta, abstractmethod
from config import get_cfg
from qa.queryUnderstanding.querySegmentation import Segmentation
from lmdb_embeddings.exceptions import MissingWordError
import numpy as np


class Embedding(metaclass=ABCMeta):
    def __init__(self, cfg):
        self.cfg = cfg
        self.seg = Segmentation(self.cfg)

    @staticmethod
    def wam(sentence, w2v_model, USE_LMDB=True, agg='mean'):
        '''
        @description: 通过word average model 生成句向量
        @param {type}
        sentence: 以空格分割的句子
        w2v_model: word2vec模型
        @return:
        '''
        arr = []
        for s in sentence:
            if USE_LMDB:
                try:
                    arr.append(w2v_model.get_word_vector(s))
                except MissingWordError:
                    pass
            else:
                if s in w2v_model.wv.vocab.keys():
                    arr.append(w2v_model.wv.get_vector(s))
        if agg == 'mean':
            return np.mean(np.array(arr), axis=0)
        elif agg == 'sum':
            return np.sum(np.array(arr), axis=0)
        else:
            raise NotImplementedError
    
    @abstractmethod
    def get_embedding_helper(self):
        pass