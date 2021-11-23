from config import get_cfg
from qa.tools import Singleton, setup_logger
from qa.tools.ahocorasick import Ahocorasick
from qa.queryUnderstanding.querySegmentation import Words
from functools import reduce

from qa.tools.utils import flatten
logger = setup_logger()


@Singleton
class Normalization():
    def __init__(self, cfg):
        logger.info('Initializing Normalizing Object ....')
        self.cfg = cfg
        self.ah = Ahocorasick()
        self.__specialize = Words(cfg).get_specializewords
        self.get_synonym_dict()
        self.build_ahocorasick()

    def is_disease(self, s):
        return s in self.__disease
    
    def is_symptom(self, s):
        return s in self.__symptom

    def is_dog(self, s):
        return s in self.__dog

    def is_cat(self, s):
        return s in self.__cat

    def get_alias(self, x):
        return self.__name2alias.get(x, "")

    def get_name(self, x):
        return self.__alias2name.get(x, "")

    def get_class(self, x):
        return self.__name2class.get(x, "")

    def get_synonym_dict(self):
        self.__disease = list(self.__specialize['疾病'].keys())
        self.__symptom = list(self.__specialize['症状'].keys())
        self.__dog = list(self.__specialize['犬'].keys())
        self.__cat = list(self.__specialize['猫'].keys())
        self.__name2class = {n: c for c, b in self.__specialize.items() for n, _ in b.items()}
        self.__name2alias = reduce(lambda a, b: dict(a, **b), self.__specialize.values())
        self.__alias2name = {j if j else k: k for k, v in self.__name2alias.items() for j in v}

    def build_ahocorasick(self):
        all_word = list(
            set([
                y for x in [[k, v] for k, v in self.__alias2name.items()]
                for y in x
            ]))
        for i in all_word:
            self.ah.add_word(i)
        self.ah.make()

    def detect(self, query):
        res = []
        for begin, end in self.ah.search_all(query):
            res.append(query[begin:end + 1])
        return res


if __name__ == '__main__':
    cfg = get_cfg()
    normalization = Normalization(cfg)
    query = '哈士奇拆家怎么办?'
    noun = normalization.detect(query)
    # noun = [x for x in noun if x in normalization.alias2name.keys()][0]
    normalize = {x: normalization.get_name(x) for x in noun}
    synonym = {k: normalization.get_alias(v) for k, v in normalize.items()}

    classes = {k: normalization.get_class(v) for k, v in normalize.items()}
    print(normalize, synonym, classes)
    logger.info({"query": query, "type": "BEST_MATCH"})
    for n, ss in synonym.items():
        for s in ss:
            if n != s:
                logger.info({"query": query.replace(n, s), "type": "BEST_MATCH"})
    # WELL_MATCH： query 扩展 + query 归一结果（类别回退）
    if classes:
        for k, v in classes.items():
            logger.info({
                "query": query.replace(k, v),
                "type": "WELL_MATCH"
            })
    else:
        logger.info({"query": query, "type": "WELL_MATCH"})
    # PART_MATCH： query 扩展 + query 归一结果（类别回退）
    if classes:
        for k, v in classes.items():
            logger.info({
                "query": query.replace(k, v),
                "type": "PART_MATCH："
            })
    else:
        logger.info({"query": query, "type": "PART_MATCH："})