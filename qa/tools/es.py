from elasticsearch import Elasticsearch
from qa.tools import Singleton, setup_logger
from qa.contentUnderstanding.content_profile import ContentUnderstanding

logger = setup_logger(name='es')
__all__ = ['ES']


@Singleton
class ES:
    __slot__ = ['cfg', 'es', 'cs']

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        logger.info(f"connect to {self.cfg.BASE.ENVIRONMENT}")
        self.es = Elasticsearch('http://{}:{}@{}:{}'.format(
            'elastic',
            cfg.ES.PWD,
            cfg.ES.HOST,
            str(cfg.ES.PORT),
        ),
                                verify_certs=False)
        self.cs = ContentUnderstanding(cfg)

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
            'es_id': x['_id'],
            'index': x["_source"]['_idx'],
            'docid': x["_source"]['question'],
            'doc': {
                'question': x["_source"]['question'],
                'answer': x["_source"]['answer'],
                'question_rough_cut':
                x["_source"]['question_rough_cut'].split(),
                'question_fine_cut': x["_source"]['question_fine_cut'].split()
            },
            'score': x['_score'],
            "tag": {
                k.lower(): v
                for k, v in self.cs.understanding(x["_source"]
                                                  ['question']).items()
                if k in ['SPECIES', 'SEX', 'AGE']
            }
        } for x in res['hits']['hits']]
        return res

    def exact_search(self, _index, _query, _rough_query, _fine_query):
        query = f"""
                {{
                    "query":{{
                        "bool": {{
                            "should": [
                                {{
                                    "term": {{
                                        "question_rough_cut.raw": "{' '.join(_rough_query)}"
                                    }}
                                }},
                                {{
                                    "term": {{
                                        "question_fine_cut.raw": "{' '.join(_fine_query)}"
                                    }}
                                }},
                                    {{
                                    "term": {{
                                        "question.keyword": "{_query}"
                                    }}
                                }}
                            ]
                        }}
                    }},
                    "size": {self.cfg.RETRIEVAL.LIMIT},
                    "_source": [ "_idx", "question", "answer", "question_rough_cut", "question_fine_cut"]
                }}
                """
        return self.search(_index, query)

    def fuzzy_search(self, _index, _row, _query: list):
        query = f"""
                {{
                "query":{{
                    "match": {{
                            "{_row}.analyzed": "{' '.join(_query)}"
                        }}
                    }},
                "size": {self.cfg.RETRIEVAL.LIMIT},
                "_source": [ "_idx", "question", "answer", "question_rough_cut", "question_fine_cut"]
                }}
                """
        return self.search(_index, query)

    def fuzzy_search_both(self, _index, _rough_query: list, _fine_query: list):
        query = f"""
                {{
                "query":{{
                    "multi_match": {{
                            "query": "{' '.join(list(set(_rough_query + _fine_query)))}{' '.join(_fine_query)}",
                            "fields": ["question_rough_cut.analyzed", "question_fine_cut.analyzed"]
                        }}
                    }},
                "size": {self.cfg.RETRIEVAL.LIMIT},
                "_source": [ "_idx", "question", "answer", "question_rough_cut", "question_fine_cut"]
                }}
                """
        return self.search(_index, query)

    def get_termvector(self, docid, index, field):
        return self.es.termvectors(
            id=docid, index=index, fields='{}.analyzed'.format(
                field))['term_vectors']['{}.analyzed'.format(field)]['terms']

    def insert_mongo(self, index, data):
        from tqdm import tqdm
        
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