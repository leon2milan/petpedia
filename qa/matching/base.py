from abc import abstractmethod, ABCMeta
from qa.queryUnderstanding.querySegmentation import Words, Segmentation

__all__ = ['Matching']


class Matching(metaclass=ABCMeta):
    __slot__ = ['cfg']

    def __init__(self, cfg):
        self.seg = Segmentation(cfg)
        self.stopwords = Words(cfg).get_stopwords

    @abstractmethod
    def get_score(self):
        pass