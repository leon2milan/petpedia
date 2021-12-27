from qa.search.search import AdvancedSearch
from config import get_cfg
from qa.tools import setup_logger
import time

logger = setup_logger(name='Search Main')
__all__ = ['Search']


class Search(object):
    __slots__ = ['cfg', 'AS']

    def __init__(self, cfg):
        logger.info('Initializing Search Object ....')
        self.cfg = cfg
        self.AS = AdvancedSearch(cfg)

    def search(self, query):
        s = time.time()
        res = self.AS.search(query)
        if self.cfg.RETRIEVAL.TIME_PERFORMENCE:
            logger.info('Search takes: {}'.format(time.time() - s))
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
        '犬细小病毒的症状', '犬细小', '我和的', '阿提桑诺曼底短腿犬', '我想养个哈士奇，应该注意什么？',
        '我家猫拉稀了， 怎么办', '我家猫半夜瞎叫唤，咋办？', '猫骨折了', '狗狗装义肢', '大型犬常见病',
        '我想养个狗，应该注意什么？', '我想养个猫，应该注意什么？', '藏獒得怎么样训练才会“认主人”'
    ]
    import cProfile, pstats, io

    pr = cProfile.Profile()
    pr.enable()
    for i in test:
        start = time.time()
        res = searchObj.search(i)
        logger.info('search takes time: {}'.format(time.time() - start))
        print(i, [(x['doc']['question'], x['score']) for x in res])
        json.loads(json.dumps({"result": res}))
    
    pr.disable()
    s = io.StringIO()
    sortby = "cumtime"  # 仅适用于 3.6, 3.7 把这里改成常量了
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.dump_stats('./profile.txt')
    print(s.getvalue())
