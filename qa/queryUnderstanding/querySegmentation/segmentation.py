import jieba_fast as fjieba
import jieba_fast.posseg as pseg
from qa.tools import Singleton, setup_logger

logger = setup_logger()
__all__ = ['Segmentation']


@Singleton
class Segmentation():
    __slot__ = ['cfg', 'rough_seg', 'fine_seg']

    def __init__(self, cfg):
        self.cfg = cfg
        self.rough_seg = fjieba.Tokenizer()
        self.rough_seg.initialize()
        self.rough_seg.load_userdict(self.cfg.DICTIONARY.CUSTOM_WORDS)

        self.fine_seg = fjieba.Tokenizer()
        self.fine_seg.initialize()

        self.rough_pseg = pseg.POSTokenizer(tokenizer=self.rough_seg)
        self.fine_pseg = pseg.POSTokenizer(tokenizer=self.fine_seg)

    def cut(self, x, mode='seg', is_rough=False):
        if is_rough:
            if mode == 'pos':
                return zip(*self.rough_pseg.cut(x))
            else:
                return self.rough_seg.cut(x)
        else:
            if mode == 'pos':
                return zip(*self.fine_pseg.cut(x))
            else:
                return self.fine_seg.cut(x)

    def get_important_words(self, text, is_rough=False):
        # TODO ADD word rank function
        pass
