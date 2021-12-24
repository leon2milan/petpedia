import re
from functools import reduce
import pandas as pd
from config import get_cfg
from pypinyin import lazy_pinyin as pinyin
from qa.queryUnderstanding.querySegmentation import Words
from qa.tools.mongo import Mongo
from functools import reduce
from tqdm.notebook import tqdm
from qa.tools.ahocorasick import Ahocorasick

tqdm.pandas(desc="Data Process")


class DuringGuide:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.mongo = Mongo(cfg, cfg.BASE.QA_COLLECTION)
        self.specialize = Words(cfg).get_specializewords
        self.ah = Ahocorasick()

    @staticmethod
    def generate_prefix(s, with_pinyin=False, max_prefix_length=5):
        res = []
        for i in range(len(s)):
            if i <= max_prefix_length:
                res.append((s[:i + 1], s))
                if with_pinyin:
                    res.append(("".join(pinyin(s[:i + 1])), s))
        return res

    def build(self):
        qa = pd.DataFrame(
            list(self.mongo.find(self.cfg.BASE.QA_COLLECTION, {})))
        qa['len'] = qa['question'].apply(len)

        query = qa[(qa['len'] <= 12) & (qa['len'] >= 3)]['question'].tolist()
        merge_all = reduce(lambda a, b: dict(a, **b), self.specialize.values())
        entity = pd.DataFrame(
            [re.sub(r'\（.*\）', '', x) for x in merge_all.keys() if x] +
            [x for y in merge_all.values()
             for x in y if x]).drop_duplicates().reset_index(drop=True).values.tolist()
        entity = [x for y in entity for x in y if x]
        prefix = []
        for word in entity:
            prefix.extend(DuringGuide.generate_prefix(word, with_pinyin=True))
        for sent in query:
            prefix.extend(DuringGuide.generate_prefix(sent))

        query_suggest = pd.DataFrame(prefix, columns=['sub_query', 'query'])
        query_suggest = query_suggest.groupby('sub_query')['query'].apply(
            list).reset_index(drop=False)
        query_suggest = query_suggest[
            ~query_suggest['sub_query'].str.startswith(
                ('\"', '#', '*', '0', '1', '2', ',', '3', '，', '4', '5', '6',
                 '7', '8', '9', '?', '@', '（', 'A', 'B', 'C', 'D', 'E', 'F',
                 'G', 'H', 'I', 'J', 'K', "L", "M", 'N', 'O', 'P', 'Q', 'U',
                 'V', 'W', 'R', 'S', 'T', 'X', 'Y', 'Z', '[', '_', "“", "《",
                 '〖', '〔', '【'))]

        self.mongo.clean(self.cfg.BASE.QUERY_DURING_GUIDE)
        self.mongo.insert_many(self.cfg.BASE.QUERY_DURING_GUIDE,
                               query_suggest.to_dict('record'))


if __name__ == '__main__':
    cfg = get_cfg()
    dg = DuringGuide(cfg)
    dg.build()
