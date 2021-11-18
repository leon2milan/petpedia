import hashlib
import math
import traceback

import pandas as pd
from config import get_cfg
from core.queryUnderstanding.preprocess.preprocess import (clean,
                                                           normal_cut_sentence)
from core.queryUnderstanding.querySegmentation import Segmentation, Words
from core.tools.logger import setup_logger
from core.tools.mongo import Mongo
from tqdm import tqdm

logger = setup_logger()


class Index(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self._docid_name_map = {}  # 正排信息
        # self._docid_name_map[docid]["doc"]
        # self._docid_name_map[docid]["attr"]
        # 编号	单词（item）	倒排列表(list<(id; tf; <pos> )>);
        self._index = {}  # 倒排信息, {"hash(term)": set(doc1, doc2, doc3, ...)}
        self._all_doc_count = 0
        self._docname_docid = {}  # "filename": [id1, id2, id3 ]
        self._invalid_docid = {}

        self.seg = Segmentation(self.cfg)
        self.stopwords = Words(self.cfg).get_stopwords
        self.mongo = Mongo(self.cfg, self.cfg.INVERTEDINDEX.DB_NAME)
        self.load()

    def get_rindex(self, termhash, is_rough):
        attr = 'rough' if is_rough else 'fine'
        res = self._index.get(termhash, None)
        if res is not None:
            res = res[attr]
        return res
        
    def get_tf(self, termhash, docid, is_rough):
        attr = 'rough_attr' if is_rough else 'fine_attr'
        return self._docid_name_map[docid][attr][termhash][self.cfg.BASE.KEY_TF_INDEX]
    
    def get_ld(self, termhash, docid, is_rough):
        attr = 'rough_attr' if is_rough else 'fine_attr'
        return self._docid_name_map[docid][attr][termhash][self.cfg.BASE.KEY_LD_INDEX]

    def get_pos(self, termhash, docid, is_rough):
        attr = 'rough_attr' if is_rough else 'fine_attr'
        return self._docid_name_map[docid][attr][termhash][self.cfg.BASE.KEY_POS_INDEX]

    def get_doc_num(self):
        return self._all_doc_count

    def get_rough_l(self):
        return self._rough_avg_l
    
    def get_fine_l(self):
        return self._fine_avg_l

    def update_index(self, row):
        # row = row['question', 'answer']
        docid, is_new_doc = self.__get_docid_by_name(row)
        if is_new_doc:
            self._all_doc_count += 1
        self.__build_one_page(docid, row)
        self.__update_one_row(docid)

    def __update_one_row(self, docid):
        self.mongo.insert_one(
            'revertedIndex', {
                "idx": docid,
                "doc": self._docid_name_map[docid]['doc'],
                "fine_attr": self._docid_name_map[docid]['fine_attr'],
                "rough_attr": self._docid_name_map[docid]['rough_attr']
            })
        
        self.mongo.insert_one('forwardIndex', {
                "_id": docid,
                "doc": self._index[docid]
            })

        self.mongo.insert_one('doc2ix', {
            "_id": docid,
            "doc": self._docname_docid[docid]
        })

    def delete_index(self, query):
        if query not in self._docname_docid:
            return
        for i in self._docname_docid[query]:
            self._invalid_docid[i] = 1
        self.mongo.delete('doc2ix',
                          "idx=={}".format(self._docname_docid[query]))
        del self._docname_docid[query]

    def __get_docid_by_name(self, row):
        is_new_doc = False
        docbasename = row['question']

        if docbasename in self._docname_docid:
            self._invalid_docid[self._docname_docid[docbasename][-1]] = 1
        if docbasename not in self._docname_docid:
            self._docname_docid[docbasename] = []
            is_new_doc = True

        self._docname_docid[docbasename].append(row['index'])
        self._docname_docid[docbasename][-1] = "%s:%s" % (
            docbasename, self._docname_docid[docbasename][-1])
        return self._docname_docid[docbasename][-1], is_new_doc

    def __index_list(self):
        data = pd.DataFrame(list(self.mongo.find(self.cfg.BASE.QA_COLLECTION, {}))).dropna()

        self._fine_avg_l = data['question_fine_cut'].apply(
            len).sum() / data.shape[0]
        self._rough_avg_l = data['question_rough_cut'].apply(
            len).sum() / data.shape[0]
        self._all_doc_count = data.shape[0]
        return data.reset_index()

    def __extend_terms(self, terms):
        if terms is None:
            return []
        count = len(terms)
        i = 0
        result = {}
        while i < count - 4:
            result[terms[i + 1] + terms[i]] = 1
            result[terms[i] + terms[i + 1]] = 1
            i += 1
        return result.keys()

    def __build_one_page_attr(self, row):
        # 构建正排信息
        return row[['question', 'answer']].to_dict()

    def __build_one_page_index(self, docid, row, is_rough=True):
        # 构建倒排
        if is_rough:
            if self.cfg.INVERTEDINDEX.USE_ANSWER:
                terms = [
                    x for line in [row['question_rough_cut']] +
                    row['answer_rough_cut'] for x in line
                ]
            else:
                terms = row['question_rough_cut']
        else:
            if self.cfg.INVERTEDINDEX.USE_ANSWER:
                terms = [
                    x for line in [row['question_fine_cut']] +
                    row['answer_fine_cut'] for x in line
                ]
            else:
                terms = row['question_fine_cut']

        terms += self.__extend_terms(terms)
        logger.debug('terms: {}'.format(terms))
        # [query][词位置]
        # [query][TF打分]
        docattr = {}

        try:
            ## term 列表 转换为 词频
            term_cnt = {}
            for c in terms:
                if c not in term_cnt:
                    term_cnt[c] = 0
                term_cnt[c] += 1

            ## 倒排
            attr = 'rough' if is_rough else 'fine'
            for c in term_cnt:
                cc = hashlib.md5(c.encode(encoding='utf-8')).hexdigest()
                if cc not in self._index:
                    self._index[cc] = {}
                    self._index[cc]['rough'] = set()
                    self._index[cc]['fine'] = set()
                self._index[cc][attr].add(docid)

            # 初始化docattr
            for c in terms:
                cc = hashlib.md5(c.encode(encoding='utf-8')).hexdigest()
                if cc not in docattr:
                    docattr[cc] = {}
                    docattr[cc][self.cfg.BASE.KEY_TF_INDEX] = 0
                    docattr[cc][self.cfg.BASE.KEY_POS_INDEX] = []
                    docattr[cc][self.cfg.BASE.KEY_LD_INDEX] = 0

            # 词位置, query长度
            for term_pos, c in enumerate(terms):
                cc = hashlib.md5(c.encode(encoding='utf-8')).hexdigest()
                docattr[cc][self.cfg.BASE.KEY_POS_INDEX].append(term_pos)
                docattr[cc][self.cfg.BASE.KEY_LD_INDEX] = len(terms)

            # 计算 TF
            PAGE_TERM_COUNT = 1.0 * len(terms)
            K = 2.0 * (1 - 0.75 + 0.75 * PAGE_TERM_COUNT / 300.0)
            for c in term_cnt:
                cc = hashlib.md5(c.encode(encoding='utf-8')).hexdigest()
                tf = term_cnt[c] / PAGE_TERM_COUNT
                tf = math.sqrt(tf)
                tf = ((2.0 + 1) * tf) / (K + tf)
                docattr[cc][self.cfg.BASE.KEY_TF_INDEX] = tf

            # TODO:
            # term weight

            # 根据短语长度调整 tf
            for c in term_cnt:
                cc = hashlib.md5(c.encode(encoding='utf-8')).hexdigest()
                if len(c) > 5:
                    docattr[cc][
                        self.cfg.BASE.KEY_TF_INDEX] = self.__change_score(
                            docattr[cc][self.cfg.BASE.KEY_TF_INDEX],
                            0.01 * len(c))

        except Exception as e:
            logger.info(traceback.format_exc())
        return docattr

    def __build_one_page(self, docid, row):
        if row['question'] == "" or row['question'] is None:
            return

        if docid not in self._docid_name_map:
            self._docid_name_map[docid] = {}
        # 构建正排信息
        self._docid_name_map[docid]["doc"] = self.__build_one_page_attr(row)

        # 构建倒排信息
        self._docid_name_map[docid]["fine_attr"] = self.__build_one_page_index(
            docid, row, is_rough=False)

        self._docid_name_map[docid][
            "rough_attr"] = self.__build_one_page_index(docid,
                                                        row,
                                                        is_rough=True)

    def __change_score(self, score, weight):
        # 根据新的 weight 因子, 调整 score
        diff = 0
        w = weight
        if w == 0:
            return score
        if w > 0:
            ratio = (1.0 - score) / 1.0
            ratio = ratio * score
            diff = (1.0 - score) * ratio
            w = w + 1.0

        if w < 0:
            ratio = (score) / 1.0
            ratio = ratio * score
            diff = (0 - score) * ratio
            w = w - 1.0
        w = 1.0 - 1.0 / w

        y = score + diff * w
        return y

    def build_index(self):
        logger.info('Starting building index ...')
        self.mongo.clean('config')
        self.mongo.clean('forwardIndex')
        self.mongo.clean('revertedIndex')
        self.mongo.clean('invalid_doc')
        self.mongo.clean('doc2id')
        docs = self.__index_list()
        for _, row in docs.iterrows():
            docid, _ = self.__get_docid_by_name(row)
            self.__build_one_page(docid, row)

    def load(self):
        tmp = [x for x in self.mongo.find('config', {})]
        try:
            self._fine_avg_l = tmp[0]['value']
            self._rough_avg_l = tmp[1]['value']
            self._all_doc_count = tmp[2]['value']
            for x in self.mongo.find('forwardIndex', {}):
                self._index[x['_id']] = x['doc']

            for x in self.mongo.find('revertedIndex', {}):
                self._docid_name_map[x['_id']] = {}
                self._docid_name_map[x['_id']]['doc'] = x['doc']
                self._docid_name_map[x['_id']]['fine_attr'] = x['fine_attr']
                self._docid_name_map[x['_id']]['rough_attr'] = x['rough_attr']
            
            for x in self.mongo.find('invalid_doc', {}):
                self._invalid_docid[x['_id']] = 1
            
            for x in self.mongo.find('doc2ix', {}):
                self._docname_docid[x['_id']] = x['doc']
        except Exception as e:
            logger.info(traceback.format_exc())
            self.build_index()
            self.save()

    def save(self):
        logger.info('Starting saving ...')
        self.mongo.insert_one('config', {
            '_id': 0,
            'name': 'fine_avg_l',
            'value': self._fine_avg_l
        })
        self.mongo.insert_one('config', {
            '_id': 1,
            'name': 'rough_avg_l',
            'value': self._rough_avg_l
        })
        self.mongo.insert_one('config', {
            '_id': 2,
            'name': 'all_doc_count',
            'value': self._all_doc_count
        })
        
        for key, _ in tqdm(self._index.items()):
            self.mongo.insert_one('forwardIndex', {
                "_id": key,
                "doc": {k: list(v) for k, v in self._index[key].items()}
            })

        for key, _ in tqdm(self._invalid_docid.items()):
            self.mongo.insert_one('invalid_doc', {
                "_id": key
            })

        for key, _ in tqdm(self._docname_docid.items()):
            self.mongo.insert_one('doc2id', {
                "_id": key,
                "doc": self._docname_docid[key]
            })

        for key, _ in tqdm(self._docid_name_map.items()):
            self.mongo.insert_one(
                'revertedIndex', {
                    "_id": key,
                    "doc": self._docid_name_map[key]['doc'], 
                    "fine_attr": self._docid_name_map[key]['fine_attr'],
                    "rough_attr": self._docid_name_map[key]['rough_attr']
                })


if __name__ == "__main__":
    cfg = get_cfg()
    ri = Index(cfg)
