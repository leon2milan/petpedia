from numpy.lib.function_base import extract
from qa.queryUnderstanding.representation import W2V
from qa.queryUnderstanding.querySegmentation import Segmentation, Words
from qa.queryUnderstanding.queryReformat.queryNormalization.normalize import Normalization
from qa.tools.logger import setup_logger
from config import get_cfg
from qa.tools.utils import Singleton

logger = setup_logger(name='entity_linking')

@Singleton
class EntityLink(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.w2v = W2V(self.cfg, is_rough=True)
        self.normalization = Normalization(self.cfg)
        self.seg = Segmentation(self.cfg)

    def entity_recall(self, query):
        normalize = self.normalization.detect(query)
        logger.debug(f"query: {query}, normalize: {normalize}")
        normalize = [self.normalization.get_name(x) for x in normalize]
        logger.debug(f"query: {query}, normalize1: {normalize}")
        normalize = [(x, self.normalization.get_class(x)) for x in normalize if x]
        logger.debug(f"query: {query}, normalize2: {normalize}")
        return normalize

    def extract_info(self):
        pass

    def entity_link(self, query):
        # candidate = self.knowledge_hnsw.search(self.seg(query, is_rough=True))
        extracted = self.entity_recall(query)
        logger.debug(f"query: {query}, extracted: {extracted}")
        # candidate = [x if x['entity'] in extracted else x['score'] * 0.5 for x in candidate]
        # candidate = sorted(candidate, key=lambda x: x['score'])
        disease = [x for x in extracted if self.normalization.is_disease(x[0])]
        logger.debug(f"query: {query}, disease: {disease}")
        if len(disease) > 0:
            return disease[0]
        symptom = [x for x in extracted if self.normalization.is_symptom(x[0])]
        logger.debug(f"query: {query}, symptom: {symptom}")
        if len(symptom) > 0:
            return symptom[0]
        return extracted[0] if extracted else ('', None)


if __name__ == '__main__':
    cfg = get_cfg()
    el = EntityLink(cfg)
    queries = [
        "狗乱吃东西怎么办", "边牧偶尔尿血怎么办", "猫咪经常拉肚子怎么办", "哈士奇拆家怎么办", "英短不吃东西怎么办？",
        "拉布拉多和金毛谁聪明", "折耳怀孕不吃东西怎么办？"
    ]
    for query in queries:
        print(query, el.entity_link(query))