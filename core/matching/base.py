from abc import abstractmethod, ABCMeta
from core.queryUnderstanding.querySegmentation import Words, Segmentation


class Matching(metaclass=ABCMeta):
    def __init__(self, cfg):
        self.seg = Segmentation(cfg)
        self.stopwords = Words(cfg).get_stopwords

    @abstractmethod
    def get_score(self):
        pass