from config import get_cfg
from qa.tools import setup_logger

logger = setup_logger()


class TermRetrieval():
    def __init__(self, cfg):
        self.cfg = cfg
        if self.cfg.RETRIEVAL.USE_ES:
            from qa.tools.es import ES
            self.es = ES(cfg)

    def search(self, query, mode, is_rough):
        if mode == 'BEST_MATCH':
            if self.cfg.RETRIEVAL.USE_ES:
                return self.__seek_es(query, is_rough, is_exact=True)

        elif mode == 'WELL_MATCH':
            if self.cfg.RETRIEVAL.USE_ES:
                return self.__seek_es(query, is_rough)

        elif mode == 'PART_MATCH':
            if self.cfg.RETRIEVAL.USE_ES:
                return self.__seek_es(query, is_rough)
        else:
            raise NotImplementedError

    def __seek_es(self, term, is_rough, is_exact=False):
        row = 'question_rough_cut' if is_rough else 'question_fine_cut'
        # term = " ".join([x[0] for x in term]) if isinstance(term, list) else term
        if is_exact:
            return self.es.exact_search(self.cfg.RETRIEVAL.ES_INDEX,
                                        'question', term)
        else:
            return self.es.fuzzy_search(self.cfg.RETRIEVAL.ES_INDEX, row, term)


if __name__ == '__main__':
    cfg = get_cfg()
    tr = TermRetrieval(cfg)
