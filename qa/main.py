from qa.search.search import AdvancedSearch
from config import get_cfg
from qa.tools import setup_logger
import time

logger = setup_logger(name='Search Main')


class Search(object):
    def __init__(self, cfg):
        logger.info('Initializing Search Object ....')
        self.cfg = cfg
        self.AS = AdvancedSearch(cfg)

    def search(self, query):
        res = self.AS.search(query)
        return [{
            k: v
            for k, v in item.items() if k in ['index', 'score', 'doc']
        } for item in res]


if __name__ == "__main__":
    cfg = get_cfg()
    searchObj = Search(cfg)
    start = time.time()
    # res = searchObj.search('哈士奇老拆家怎么办')
    # res = searchObj.search('犬瘟热')
    res = searchObj.search('狗发烧')
    print(res[0])
    # print([(x['docid'], x['score']) for x in res])
    logger.info('search takes time: {}'.format((time.time() - start) / 60))
