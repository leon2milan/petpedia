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
    import json
    cfg = get_cfg()
    searchObj = Search(cfg)
    start = time.time()
    test = ['哈士奇老拆家怎么办', '犬瘟热', '狗发烧', '金毛', '拉布拉多不吃东西怎么办', '犬细小病毒的症状', '犬细小']
    for i in test:
        res = searchObj.search(i)
        logger.info('search takes time: {}'.format((time.time() - start) / 60))
        print([(x['doc']['question'], x['score']) for x in res])
        json.loads(json.dumps({"result": res}))
