from elasticsearch import Elasticsearch
import re
from qa.tools import Singleton

@Singleton
class ES:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.es = Elasticsearch('http://{}:{}@{}:{}'.format(
            'elastic',
            cfg.ES.PWD,
            cfg.ES.HOST,
            str(cfg.ES.PORT),
        ),
                                verify_certs=False)

    def insert(self, data: dict, _index: str = "qa_v1"):
        res = self.es.index(index=_index, document=data)

    def insert_many(self, data: list, _index: str = "qa_v1"):
        for item in data:
            res = self.es.index(index=_index, document=item)
        self.es.indices.refresh(index=_index)

    def search(self, _index, _query):
        res = self.es.search(index=_index, body=_query)
        hits = res['hits']['total']['value']
        res = [{
            'es_id':
            x['_id'],
            'mongo_id':
            x["_source"]['_idx'],
            'docid':
            x["_source"]['question'],
            'doc':{
                'question': x["_source"]['question'],
                'answer': x["_source"]['answer']
            },
            'score':
            x['_score'],
            'hit_words': [
                "".join(
                    re.findall(re.compile(u"[\u4e00-\u9fa5]+"),
                               y['description']))
                for y in x['_explanation']['details']
            ]
        } for x in res['hits']['hits']]
        return res

    def exact_search(self, _index, _row, _query):
        query = f"""
                {{
                "query":{{
                    "term": {{
                            "{_row}": "{_query}"
                        }}
                    }},
                "size": {self.cfg.RETRIEVAL.LIMIT},
                "_source": [ "_idx", "question", "answer"],
                "explain": "true"
                }}
                """
        return self.search(_index, query)

    def fuzzy_search(self, _index, _row, _query):
        query = f"""
                {{
                "query":{{
                    "match": {{
                            "{_row}.analyzed": "{' OR '.join(_query.split(' '))}"
                        }}
                    }},
                "size": {self.cfg.RETRIEVAL.LIMIT},
                "_source": [ "_idx", "question", "answer"],
                "explain": "true"
                }}
                """
        return self.search(_index, query)

    def get_termvector(self, docid, index, field):
        return self.es.termvectors(
            id=docid, index=index, fields='{}.analyzed'.format(
                field))['term_vectors']['{}.analyzed'.format(field)]['terms']

    def insert_mongo(self, index):
        from qa.tools.mongo import Mongo
        import pandas as pd
        from tqdm import tqdm
        mongo = Mongo(self.cfg, self.cfg.INVERTEDINDEX.DB_NAME)
        data = pd.DataFrame(list(mongo.find(self.cfg.BASE.QA_COLLECTION,
                                            {}))).dropna()
        data['_id'] = data['_id'].apply(str)
        data['question_fine_cut'] = data['question_fine_cut'].progress_apply(
            lambda x: " ".join(x))

        data['question_rough_cut'] = data['question_rough_cut'].progress_apply(
            lambda x: " ".join(x))

        data = data[[
            'question_fine_cut', 'question_rough_cut', 'answer', "_id",
            "question"
        ]]
        data.columns = [
            'question_fine_cut', 'question_rough_cut', 'answer', "_idx",
            "question"
        ]
        for idx, row in tqdm(data.iterrows(), total=data.shape[0]):
            tmp = row.to_dict()
            self.es.index(index, tmp)
        self.es.indices.refresh(index=index)