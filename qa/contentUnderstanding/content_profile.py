import pandas as pd
from qa.tools.mongo import Mongo
from qa.queryUnderstanding.preprocess import clean
from qa.queryUnderstanding.querySegmentation import Segmentation, Words
from config import get_cfg
from functools import reduce
from tqdm.notebook import tqdm
import toml
from qa.knowledge.entity_link import EntityLink

tqdm.pandas(desc="Data Process")


class ContentUnderstanding:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.mongo = Mongo(self.cfg, self.cfg.INVERTEDINDEX.DB_NAME)
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
        if not isinstance(string, list):
            rough_cut = self.seg.cut(clean(string), is_rough=True)
        else:
            string = ''.join(string)
            rough_cut = string
        entity = self.el.get_mentions(string)
        breed_name = list(set([y[0] for y in entity
                               if y[1] in ['DOG', 'CAT']]))
        disease_name = list(set([y[0] for y in entity if y[1] in ['DISEASE']]))
        symptom_name = list(set([y[0] for y in entity if y[1] in ['SYMPTOM']]))
        content_tag = self.content_tag(rough_cut)
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
        content_tag['SPECIES'] = list(
            set([y[1] for y in entity if y[1] in ['DOG', 'CAT']
                 ])) if not content_tag['SPECIES'] else content_tag['SPECIES']
        content_tag['breed_name'] = breed_name
        content_tag['disease_name'] = disease_name
        content_tag['symptom_name'] = symptom_name
        return content_tag

    def save_to_mongo(self):
        qa = pd.DataFrame(list(self.mongo.find(self.cfg.BASE.QA_COLLECTION, {})))
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
    cs.save_to_mongo()
