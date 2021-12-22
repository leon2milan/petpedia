from config import get_cfg
from qa.tools import Singleton, setup_logger
from qa.tools.ahocorasick import Ahocorasick
from qa.queryUnderstanding.querySegmentation import Words
from functools import reduce
from collections import defaultdict
from qa.tools.utils import flatten

logger = setup_logger()


@Singleton
class Normalization():
    def __init__(self, cfg):
        logger.info('Initializing Normalizing Object ....')
        self.cfg = cfg
        self.ah = Ahocorasick()
        self.word = Words(cfg)
        self.build_ahocorasick()

    def build_ahocorasick(self):
        for k, item in self.word._Words__alias2name.items():
            self.ah.add_word(k)
            for i in item:
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
    normalize = {x: normalization.get_name(x) for x in noun}
    synonym = {
        k: normalization.get_alias(x)
        for k, v in normalize.items() for x in v
    }

    classes = {
        k: normalization.get_class(x)
        for k, v in normalize.items() for x in v
    }
    print(normalize, synonym, classes)
    logger.info({"query": query, "type": "BEST_MATCH"})
    for n, ss in synonym.items():
        for s in ss:
            if n != s:
                logger.info({
                    "query": query.replace(n, s),
                    "type": "BEST_MATCH"
                })
    # WELL_MATCH： query 扩展 + query 归一结果（类别回退）
    if classes:
        for k, v in classes.items():
            logger.info({"query": query.replace(k, v), "type": "WELL_MATCH"})
    else:
        logger.info({"query": query, "type": "WELL_MATCH"})
    # PART_MATCH： query 扩展 + query 归一结果（类别回退）
    if classes:
        for k, v in classes.items():
            logger.info({"query": query.replace(k, v), "type": "PART_MATCH："})
    else:
        logger.info({"query": query, "type": "PART_MATCH："})
