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
        s = time.time()
        res = self.AS.search(query)
        logger.debug('Search takes: {}'.format(time.time() - s))
        return [{
            k: v
            for k, v in item.items() if k in ['index', 'score', 'doc']
        } for item in res]


if __name__ == "__main__":
    import json
    cfg = get_cfg()
    searchObj = Search(cfg)
    test = [
        '狗狗容易感染什么疾病', '哈士奇老拆家怎么办', '犬瘟热', '狗发烧', '金毛', '拉布拉多不吃东西怎么办',
        '犬细小病毒的症状', '犬细小', '我和的', '阿提桑诺曼底短腿犬'
    ]
    for i in test:
        start = time.time()
        res = searchObj.search(i)
        logger.info('search takes time: {}'.format(time.time() - start))
        print([(x['doc']['question'], x['score']) for x in res])
        json.loads(json.dumps({"result": res}))
