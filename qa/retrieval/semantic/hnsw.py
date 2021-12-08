# -*- coding: UTF-8 -*-
import hnswlib
import os
import pickle
import numpy as np
import pandas as pd
from functools import partial
from qa.queryUnderstanding.querySegmentation import Segmentation, Words
from qa.queryUnderstanding.representation import REPRESENTATION_REGISTRY
from qa.tools.mongo import Mongo
from qa.tools.logger import setup_logger
from config import get_cfg
import gc
from abc import ABCMeta, abstractmethod

from qa.tools.utils import Singleton

logger = setup_logger()
__all__ = ["ANN", "HNSW"]


class ANN(metaclass=ABCMeta):
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def data_load(self):
        pass

    @abstractmethod
    def search(self):
        pass

@Singleton
class HNSW(ANN):
    __type = "hnsw"

    def __init__(self, cfg, is_rough) -> None:
        self.cfg = cfg
        self.is_rough = is_rough
        self.seg = Segmentation(cfg)
        self.stopwords = Words(cfg).get_stopwords
        self.mongo = Mongo(self.cfg, self.cfg.INVERTEDINDEX.DB_NAME)

        logger.info('Loading Vector based retrieval model {} ....'.format(
            'rough' if is_rough else 'fine'))
        self.embedding = REPRESENTATION_REGISTRY.get(
            self.cfg.RETRIEVAL.HNSW.SENT_EMB_METHOD)(self.cfg,
                                                     is_rough=self.is_rough)
        self.sent_func = self.embedding.get_embedding_helper
        self.emb_size = self.cfg.REPRESENTATION.WORD2VEC.EMBEDDING_SIZE if 'BERT' not in self.cfg.RETRIEVAL.HNSW.SENT_EMB_METHOD else self.cfg.REPRESENTATION.BERT.EMBEDDING_SIZE

        self.hnsw = self.load(is_rough=self.is_rough)

    def train(self, data, to_file: str, ids=None):
        '''
        @description: 训练hnsw模型
        @param {type}
        to_file： 模型保存目录
        ef/m 等参数 参考https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
        @return:
        '''
        dim = self.emb_size
        assert dim == data.shape[1]
        hnsw = data
        num_elements = data.shape[0]
        assert num_elements == max(ids) + 1, f"num_elements: {num_elements}, max id: {max(ids) + 1}"

        # Declaring index
        p = hnswlib.Index(space=self.cfg.RETRIEVAL.HNSW.SPACE,
                          dim=dim)  # possible options are
        p.init_index(max_elements=num_elements,
                     ef_construction=self.cfg.RETRIEVAL.HNSW.EF_CONSTRUCTION,
                     M=self.cfg.RETRIEVAL.HNSW.M)
        p.set_ef(self.cfg.RETRIEVAL.HNSW.EF)
        p.set_num_threads(self.cfg.RETRIEVAL.HNSW.THREAD)
        if ids is not None:
            p.add_items(hnsw, ids)
        else:
            p.add_items(hnsw)
        logger.info('Start')
        labels, distances = p.knn_query(hnsw, k=1)
        logger.info(
            f"Parameters passed to constructor:  space={p.space}, dim={p.dim}")
        logger.info(
            f"Index construction: M={p.M}, ef_construction={p.ef_construction}"
        )
        logger.info(
            f"Index size is {p.element_count} and index capacity is {p.max_elements}"
        )
        logger.info(f"Search speed/quality trade-off parameter: ef={p.ef}")
        logger.info(" Recall:{}".format(
            np.mean(labels.reshape(-1) == np.arange(len(hnsw)))))
        test_vec = self.sent_func(['哈士奇', '拆家'])
        labels, distances = p.knn_query(test_vec, k=10)
        # print(labels, distances)
        # p.save_index(to_file)
        pickle.dump(p, open(to_file, 'wb'))
        return p

    def data_load(self, is_rough=False):
        '''
        @description: 读取数据， 并生成句向量
        @param {type}
        data_path：问答pair数据所在路径
        @return: 包含句向量的dataframe
        '''
        logger.info('Loading training data ....')
        data = pd.DataFrame(
            list(self.mongo.find(self.cfg.BASE.QA_COLLECTION, {})))

        col = 'question_rough_cut' if is_rough else 'question_fine_cut'
        data['embedding'] = data[col].progress_apply(
            lambda x: self.sent_func(x))
        data['embedding'] = data['embedding'].progress_apply(
            lambda x: x
            if x.shape[1] == self.emb_size else np.zeros((1, self.emb_size)))
        return data[['index', 'embedding']]

    def load_helper(self, model_path):
        '''
        @description: 加载训练好的hnsw模型
        @param {type}
        model_path： 模型保存的目录
        @return: hnsw 模型
        '''
        logger.info('Loading hnsw model...')
        # hnsw = hnswlib.Index(space=self.cfg.RETRIEVAL.HNSW.SPACE, dim=self.emb_size)
        # hnsw.load_index(model_path, max_elements=num_elements)
        hnsw = pickle.load(open(model_path, 'rb'))
        return hnsw

    def load(self, is_rough=False):
        path = self.cfg.RETRIEVAL.HNSW.ROUGH_HNSW_PATH if is_rough else self.cfg.RETRIEVAL.HNSW.FINE_HNSW_PATH

        if not self.cfg.RETRIEVAL.HNSW.FORCE_TRAIN and os.path.exists(path):
            # 加载
            return self.load_helper(path)
        else:
            # 训练
            data = self.data_load(is_rough)
            return self.train(
                np.stack(data['embedding'].values).reshape(-1, self.emb_size),
                path, data['index'].values)

    def search(self, qeury_list):
        test_vec = self.sent_func(qeury_list)
        if test_vec.shape != (1, self.emb_size):
            return []
        labels, distances = self.hnsw.knn_query(test_vec,
                                                k=int(
                                                    self.cfg.RETRIEVAL.LIMIT))
        distances = [
            x for x in distances.reshape(-1).tolist()
            if x < self.cfg.RETRIEVAL.HNSW.FILTER_THRESHOLD
        ]
        labels = labels.reshape(-1).tolist()[:len(distances)]
        distances = dict(zip(labels, distances))
        # note: mongo find in return shuffled data. 
        res = [{
            'docid':
            item['question'] if self.cfg.RETRIEVAL.USE_ES else
            item['question'] + ":" + str(item['index']),
            'index':
            str(item['_id']),
            'score':
            20 - distances[item['index']],
            'pos': [],
            "doc": {
                'question': item['question'],
                'answer': item['answer'],
            },
            "source":
            'rough' if self.is_rough else 'fine',
        } for item in self.mongo.find(self.cfg.BASE.QA_COLLECTION,
                                      {'index': {
                                          "$in": labels
                                      }})] 
        return sorted(res, key=lambda key: key['score'], reverse=True)


if __name__ == "__main__":
    cfg = get_cfg()
    rough = HNSW(cfg, is_rough=True)
    print('rough', [(x['docid'], x['score'], x['index'])
                    for x in rough.search(['哈士奇', '拆家'])])

    fine = HNSW(cfg, is_rough=False)
    print('fine', [x['docid'] for x in fine.search(['哈士奇', '拆', '家'])])
    print('fine', [x['docid'] for x in fine.search(['狗', '老', '拆', '家'])])
    print('fine', [x['docid'] for x in fine.search(['哈士奇', '老', '拆', '家'])])
    print('fine', [x['docid'] for x in rough.search(['犬细小'])])

    # know = KNOWLEDGE_ANN(cfg)
    # print('know', [x['entity'] for x in know.search(['西伯利亚哈士奇犬', '拆家'])])
    # print('know', [x['entity'] for x in know.search(['哈士奇', '拆', '家'])])
    # print('know', [x['entity'] for x in know.search(['狗', '老', '拆', '家'])])
    # print('know', [x['entity'] for x in know.search(['哈士奇', '老', '拆', '家'])])