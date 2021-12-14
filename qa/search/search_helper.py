from qa.tools.mongo import Mongo
from config import get_cfg
from qa.queryUnderstanding.querySegmentation import Words
from qa.tools.ahocorasick import Ahocorasick


class SearchHelper:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.mongo = Mongo(cfg, self.cfg.INVERTEDINDEX.DB_NAME)
        self.build_sensetive_detector()

    def build_sensetive_detector(self):
        sw = Words(self.cfg).get_sensitivewords
        self.sen_detector = Ahocorasick()
        for word in sw:
            self.sen_detector.add_word(word)
        self.sen_detector.make()

    def sensetive_detect(self, query):
        res = self.sen_detector.search_all(query)
        flag = False 
        if len(res) > 0:
            flag = True
            res = [query[x[0]: x[1] + 1] for x in res]
        return flag, res

    def hot_query(self):
        res = self.mongo.find(
            self.cfg.BASE.QA_COLLECTION,
            {'index': {
                "$in": [68588, 4715, 2836, 109191, 484]
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
    print(helper.sensetive_detect('习大大'))
    print(helper.sensetive_detect('犬细小病毒的症状'))