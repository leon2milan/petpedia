from config import get_cfg
from qa.tools import setup_logger

logger = setup_logger()
__all__ = ['TermRetrieval']


class TermRetrieval():
    __slot__ = ['cfg', 'es']

    def __init__(self, cfg):
        self.cfg = cfg
        if self.cfg.RETRIEVAL.USE_ES:
            from qa.tools.es import ES
            self.es = ES(cfg)

    def search(self, query, rough_query, fine_query, mode):
        if mode == 'BEST_MATCH':
            if self.cfg.RETRIEVAL.USE_ES:
                return self.__seek_es(query,
                                      rough_query,
                                      fine_query,
                                      is_exact=True)

        elif mode == 'WELL_MATCH':
            if self.cfg.RETRIEVAL.USE_ES:
                return self.__seek_es(query, rough_query, fine_query)

        elif mode == 'PART_MATCH':
            if self.cfg.RETRIEVAL.USE_ES:
                return self.__seek_es(query)
        else:
            raise NotImplementedError

    def __seek_es(self,
                  term,
                  rough_query,
                  fine_query,
                  is_exact=False):
        # term = " ".join([x[0] for x in term]) if isinstance(term, list) else term
        if is_exact:
            return self.es.exact_search(self.cfg.RETRIEVAL.ES_INDEX, term,
                                        rough_query, fine_query)
        else:
            return self.es.fuzzy_search_both(self.cfg.RETRIEVAL.ES_INDEX,
                                             rough_query, fine_query)


if __name__ == '__main__':
    cfg = get_cfg()
    tr = TermRetrieval(cfg)
