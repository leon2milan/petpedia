from qa.tools.mongo import Mongo
from config import get_cfg
from qa.queryUnderstanding.querySegmentation import Words
from qa.tools.ahocorasick import Ahocorasick

__all__ = ['SearchHelper']


class SearchHelper:
    __slot__ = ['cfg', 'mongo', 'sen_detector']

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.mongo = Mongo(cfg, self.cfg.BASE.QA_COLLECTION)

    def hot_query(self):
        res = self.mongo.find(
            self.cfg.BASE.QA_COLLECTION,
            {'index': {
                "$in": [47040, 49111, 49706, 1344, 47131]
            }})
        res = [{
            'doc': {
                'question': x['question'],
                'answer': x['answer']
            },
            'index': x['index']
        } for x in list(res)]
        return list(res)


if __name__ == "__main__":
    cfg = get_cfg()
    helper = SearchHelper(cfg)
    print([x['doc']['question'] for x in helper.hot_query()])