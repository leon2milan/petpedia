from LAC import LAC
from core.tools import Singleton, setup_logger
logger = setup_logger()

@Singleton
class Segmentation():
    def __init__(self, cfg):
        self.cfg = cfg
        self.rough_lac_rank = LAC(mode='rank')
        self.rough_lac_rank.load_customization(self.cfg.DICTIONARY.CUSTOM_WORDS,
                                               sep=None)

        self.rough_lac_seg = LAC(mode='seg')
        self.rough_lac_seg.load_customization(self.cfg.DICTIONARY.CUSTOM_WORDS,
                                              sep=None)

        self.fine_lac_seg = LAC(mode='seg')
        self.fine_lac_rank = LAC(mode='rank')

    def cut(self, x, mode='seg', is_rough=False):
        if is_rough:
            if mode == 'seg':
                return self.rough_lac_seg.run(x)
            elif mode == 'rank':
                return self.rough_lac_rank.run(x)
            else:
                raise NotImplementedError
        else:
            if mode == 'seg':
                return self.fine_lac_seg.run(x)
            elif mode == 'rank':
                return self.fine_lac_rank.run(x)
            else:
                raise NotImplementedError

    def get_important_words(self, text, is_rough=False):
        if text.isdecimal():
            return []
        if is_rough:
            s = self.fine_lac_rank.cut(text)
        else:
            s = self.rough_lac_rank.cut(text)
        ret = [x[0] for x in zip(*s) if any([y in x[1] for y in "n"])]
        # n:名词 v:动词 r:代词(你我他啥) p:介词 e:叹词
        logger.debug([x for x in zip(*s)])
        logger.debug("分词分析重要词:{}".format(ret))
        return ret
