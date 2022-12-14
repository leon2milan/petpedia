# -*- coding: UTF-8 -*-
from time import time
import hnswlib
import os
import pickle
import numpy as np
import pandas as pd
from qa.queryUnderstanding.querySegmentation import Segmentation, Words
from qa.queryUnderstanding.representation import REPRESENTATION_REGISTRY
from qa.tools.mongo import Mongo
from qa.tools.logger import setup_logger
from config import get_cfg
import time
from abc import ABCMeta, abstractmethod

from qa.tools.utils import Singleton

logger = setup_logger()
__all__ = ["HNSW"]


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
    __slot__ = [
        'cfg', 'is_rough', 'mongo', 'embedding',
        'sent_func', 'hnsw', 'emb_size', 'id_map'
    ]

    def __init__(self, cfg, is_rough) -> None:
        self.cfg = cfg
        self.is_rough = is_rough
        self.mongo = Mongo(self.cfg, self.cfg.BASE.QA_COLLECTION)

        logger.info('Loading Vector based retrieval model {} ....'.format(
            'rough' if is_rough else 'fine'))
        self.embedding = REPRESENTATION_REGISTRY.get(
            self.cfg.RETRIEVAL.HNSW.SENT_EMB_METHOD)(self.cfg,
                                                     is_rough=self.is_rough)
        self.sent_func = self.embedding.get_embedding_helper
        self.emb_size = self.cfg.REPRESENTATION.WORD2VEC.EMBEDDING_SIZE if 'BERT' not in self.cfg.RETRIEVAL.HNSW.SENT_EMB_METHOD else self.cfg.REPRESENTATION.BERT.EMBEDDING_SIZE

        self.hnsw = self.load(is_rough=self.is_rough)
        if self.is_need_incremental_learning():
            # self.incremental_train(is_rough=self.is_rough)
            pass

        self.id_map = self.get_id_mapping()

    def get_data_count(self):
        return self.hnsw.get_current_count()

    def is_need_incremental_learning(self):
        if self.mongo.get_col_stats(
            'qa')['executionStats']['totalDocsExamined'] != self.get_data_count():
            flag = True
        else:
            flag = False
        return flag

    def get_id_mapping(self):
        qa = pd.DataFrame(
            list(self.mongo.find(self.cfg.BASE.QA_COLLECTION,
                                 {})))[['_id', 'index']]
        id_map = dict(zip(qa['index'], qa['_id']))
        return id_map

    def train(self, data, to_file: str, ids=None):
        '''
        @description: ??????hnsw??????
        @param {type}
        to_file??? ??????????????????
        ef/m ????????? ??????https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
        @return:
        '''
        dim = self.emb_size
        assert dim == data.shape[1]
        hnsw = data
        num_elements = data.shape[0]
        assert num_elements == max(
            ids) + 1, f"num_elements: {num_elements}, max id: {max(ids) + 1}"

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
        test_vec = self.sent_func(['?????????', '??????'])
        labels, distances = p.knn_query(test_vec, k=10)
        # print(labels, distances)
        # p.save_index(to_file)
        pickle.dump(p, open(to_file, 'wb'))
        return p

    def increamental_train(self, is_rough=False):
        path = self.cfg.RETRIEVAL.HNSW.ROUGH_HNSW_PATH if is_rough else self.cfg.RETRIEVAL.HNSW.FINE_HNSW_PATH

        origin_cnt = self.get_data_count()

        query = {"index": {"$gt": origin_cnt}}
        data = self.data_load(is_rough, query)
        print('data', data.shape)
        self.hnsw.add_items(np.stack(data['embedding'].values).reshape(-1, self.emb_size),
                            data['index'].values)

        test_vec = self.sent_func(['?????????', '??????'])
        labels, distances = self.hnsw.knn_query(test_vec, k=10)
        pickle.dump(self.hnsw, open(path, 'wb'))
        logger.info(f"Incremental training is Done!!!")

    def data_load(self, is_rough=False, query=None):
        '''
        @description: ??????????????? ??????????????????
        @param {type}
        data_path?????????pair??????????????????
        @return: ??????????????????dataframe
        '''
        logger.info('Loading training data ....')
        if query is not None:
            data = pd.DataFrame(
                list(self.mongo.find(self.cfg.BASE.QA_COLLECTION, {})))
        else:
            data = pd.DataFrame(
                list(self.mongo.find(self.cfg.BASE.QA_COLLECTION, {})))

        col = 'question_rough_cut' if is_rough else 'question_fine_cut'
        data['embedding'] = data[col].progress_apply(
            lambda x: self.sent_func(x).reshape(1, -1))
        data['embedding'] = data['embedding'].progress_apply(
            lambda x: x
            if x.shape[1] == self.emb_size else np.zeros((1, self.emb_size)))
        return data[['index', 'embedding']]

    def load_helper(self, model_path):
        '''
        @description: ??????????????????hnsw??????
        @param {type}
        model_path??? ?????????????????????
        @return: hnsw ??????
        '''
        logger.info('Loading hnsw model...')
        hnsw = pickle.load(open(model_path, 'rb'))
        return hnsw

    def load(self, is_rough=False):
        path = self.cfg.RETRIEVAL.HNSW.ROUGH_HNSW_PATH if is_rough else self.cfg.RETRIEVAL.HNSW.FINE_HNSW_PATH

        if not self.cfg.RETRIEVAL.HNSW.FORCE_TRAIN and os.path.exists(path):
            # ??????
            return self.load_helper(path)
        else:
            # ??????
            data = self.data_load(is_rough)
            return self.train(
                np.stack(data['embedding'].values).reshape(-1, self.emb_size),
                path, data['index'].values)

    def search(self, qeury_list):
        if not any(isinstance(el, list) for el in qeury_list):
            qeury_list = [qeury_list]
        test_vec = self.sent_func(qeury_list)
        if test_vec.shape != (len(qeury_list), self.emb_size):
            print(
                f"test_vec: {test_vec.shape} != {(len(qeury_list), self.emb_size)}"
            )
            return []
        labels, distances = self.hnsw.knn_query(test_vec,
                                                k=int(
                                                    self.cfg.RETRIEVAL.LIMIT))
        # labels = labels[np.where(distances < self.cfg.RETRIEVAL.HNSW.FILTER_THRESHOLD)]
        # distances = distances[
        #     np.where(distances < self.cfg.RETRIEVAL.HNSW.FILTER_THRESHOLD)]
        distances = dict(
            zip(labels.reshape(-1).tolist(),
                distances.reshape(-1).tolist()))
        # note: mongo find in return shuffled data.
        res = [[
            {
                'docid':
                item['question'] if self.cfg.RETRIEVAL.USE_ES else
                item['question'] + ":" + str(item['index']),
                'index':
                str(item['_id']),
                'score':
                20 - distances[item['index']] * 1 if self.is_rough else 2.0,
                'pos': [],
                "doc": {
                    'question': item['question'],
                    'answer': item['answer'],
                    'question_rough_cut': item['question_rough_cut'],
                    'question_fine_cut': item['question_fine_cut']
                },
                "source":
                'rough' if self.is_rough else 'fine',
                "tag": {
                    'species': item['SPECIES'],
                    'sex': item['SEX'],
                    'age': item['AGE'],
                }
            } for item in self.mongo.find(self.cfg.BASE.QA_COLLECTION, {
                '_id': {
                    "$in": [self.id_map[x] for x in labels[i].tolist()]
                }
            }) if
            distances[item['index']] < self.cfg.RETRIEVAL.HNSW.FILTER_THRESHOLD
        ] for i in range(len(labels))]
        return [
            sorted(x, key=lambda keys: keys['score'], reverse=True)
            for x in res
        ]


if __name__ == "__main__":
    cfg = get_cfg()
    rough = HNSW(cfg, is_rough=True)
    print('count', rough.get_data_count())
    test = [['??????', '???', '??????', '???', '??????', '??????'], ['???', '??????', '???'],
            ['??????', '??????', '?????????', '???', '??????', '????????????', '???'],
            ['??????', '??????', '??????', '??????', '??????'], ['?????????', '???', '??????'],
            [['?????????'], ['?????????', '???', '??????']], ['?????????', '??????'],
            ['??????', '??????', '???', '????????????'], ['??????', '??????', '?????????', '????????????'],
            ['??????', '??????', '???', '????????????'],
            ['??????', '???', '??????', '?????????', '???']]
    for x in test:
        s = time.time()
        print('query', x, 'rough', [[(x['docid'], x['score'], x['index'])
                                     for x in y]
                                    for y in rough.search(x)], 'takes: ',
              time.time() - s)

    fine = HNSW(cfg, is_rough=False)
    test = [['?????????', '??????'], ['???', '???', '??????', '?????????'], ['?????????', '???', '??????'],
            ['???', '??????'],
            ['??????', '???', '??????', '???', '??????', '???']]
    for x in test:
        s = time.time()
        print('fine', [[y['docid'] for y in x] for x in fine.search(x)],
              'takes: ',
              time.time() - s)
