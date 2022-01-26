import pandas as pd
import toml
from config import get_cfg
from qa.knowledge.entity_link import EntityLink
from qa.queryUnderstanding.preprocess import clean
from qa.queryUnderstanding.querySegmentation import Segmentation, Words
from qa.tools.mongo import Mongo
from tqdm.notebook import tqdm

tqdm.pandas(desc="Data Process")
__all__ = ['ContentUnderstanding']


class ContentUnderstanding:
    __slot__ = [
        'cfg', 'mongo', 'seg', 'stopwords', 'specialize', 'keyword', 'el'
    ]

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.mongo = Mongo(self.cfg, self.cfg.BASE.QA_COLLECTION)
        self.seg = Segmentation(self.cfg)
        self.stopwords = Words(self.cfg).get_stopwords
        self.specialize = Words(self.cfg).get_specializewords
        self.keyword = toml.load(self.cfg.CONTENTUNDERSTANDING.KEYWORD_FILE)
        self.el = EntityLink(self.cfg)

    def tag_helper(self, string, keyword_list):
        tmp = [x for x in string if x in keyword_list]
        return list(set(tmp))

    def content_tag(self, string):
        res = {}
        for scenario in self.keyword.keys():
            if isinstance(self.keyword[scenario], list):
                for item in self.keyword[scenario]:
                    res[item['name']] = self.tag_helper(
                        string, item['contain'])
            else:
                res[self.keyword[scenario]['name']] = self.tag_helper(
                    string, self.keyword[scenario]['contain'])
        return {k: v for k, v in res.items() if v}

    def understanding(self, string):
        entity = self.el.get_mentions(string)
        breed_name = list(set([y[0] for y in entity
                               if y[1] in ['DOG', 'CAT']]))
        disease_name = list(set([y[0] for y in entity if y[1] in ['DISEASE']]))
        symptom_name = list(set([y[0] for y in entity if y[1] in ['SYMPTOM']]))
        content_tag = self.content_tag(string)
        for key in self.keyword.keys():
            if isinstance(self.keyword[key], list):
                cols = [
                    x['name'] for x in self.keyword[key]
                    if x['name'] in content_tag.keys()
                ]
                content_tag[key] = [
                    col for col in cols for x in content_tag[col] if x
                ]
                for col in cols:
                    del content_tag[col]
            elif isinstance(self.keyword[key], dict) and key not in [
                    'SYMPTOM', 'DISEASE', 'BREED'
            ]:
                content_tag[key] = ["|".join(content_tag[key]) for x in content_tag.get(key, []) if x]

        content_tag['SPECIES'] = list(
            set([y[1] for y in entity if y[1] in ['DOG', 'CAT']
                 ])) if not content_tag['SPECIES'] else content_tag['SPECIES']
        content_tag['breed_name'] = breed_name
        content_tag['disease_name'] = disease_name
        content_tag['symptom_name'] = symptom_name
        return content_tag

    def save_to_mongo(self):
        qa = pd.DataFrame(
            list(self.mongo.find(self.cfg.BASE.QA_COLLECTION, {})))
        qa['content_tag'] = qa.apply(
            lambda row: self.understanding(row['question_rough_cut']), axis=1)
        qa = pd.concat([
            qa.drop(['content_tag'], axis=1), qa['content_tag'].apply(
                pd.Series)
        ],
                       axis=1)

        def format_content(x):
            if isinstance(x, list):
                return "|".join(x)
            return x

        for col in qa.columns:
            if col not in [
                    'question_rough_cut', 'question_fine_cut', 'index', '_id'
            ]:
                qa[col] = qa[col].fillna('').apply(lambda x: format_content(x))
        qa = qa[[
            '_id', 'index', 'question', 'answer', 'question_fine_cut',
            'question_rough_cut', 'breed_name', 'disease_name', 'symptom_name',
            'AGE', 'SPECIES', 'SEX', 'DISEASE', 'BASIC', 'ATTRIBUTE',
            'QUESTION', 'BEAUTY', 'HEALTHY', 'DOMESTICATE', 'BREED', 'PART',
            'FOOD', 'DRUG'
        ]]
        # all_keyword = flatten([y['contain'] if isinstance(x, list) else x['contain'] for x   in keyword.values() for y in x])
        # qa['last_word'] = qa.fillna('').progress_apply(lambda row: [x for x in list(set(row['question_rough_cut'] + row['question_fine_cut'])) if x not in all_keyword and x not in entity_word and x not in stopwords], axis=1)
        # sorted(Counter([y for x in qa['last_word'].values.tolist() for y in x]).items(), key=lambda x: x[1], reverse=True)

        self.mongo.clean(self.cfg.BASE.QA_COLLECTION)
        self.mongo.insert_many(self.cfg.BASE.QA_COLLECTION,
                               qa.fillna('').to_dict('record'))


if __name__ == '__main__':
    cfg = get_cfg()
    cs = ContentUnderstanding(cfg)
    # cs.save_to_mongo()

    test = [
        '狗狗容易感染什么疾病', '哈士奇老拆家怎么办', '犬瘟热', '狗发烧', '金毛', '拉布拉多不吃东西怎么办',
        '犬细小病毒的症状', '犬细小', '我和的', '阿提桑诺曼底短腿犬', '我想养个哈士奇，应该注意什么？',
        '我家猫拉稀了， 怎么办', '我家猫半夜瞎叫唤，咋办？', '猫骨折了', '狗狗装义肢', '大型犬常见病',
        '我想养个狗，应该注意什么？', '我想养个猫，应该注意什么？'
    ]

    for i in test:
        print(i, cs.understanding(i))
