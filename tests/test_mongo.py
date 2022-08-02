import unittest
from qa.tools.mongo import Mongo
from config import get_cfg
from qa.retrieval.semantic.hnsw import HNSW


class ModelTest(unittest.TestCase):

    def setUp(self):
        cfg = get_cfg()
        self.mongo = Mongo(cfg, cfg.BASE.QA_COLLECTION)
        self.files = ['qa', 'sensetiveWord', 'toneShapeCode']
        self.db = ['qa']
        self.collections = [
            'querySuggest', 'sensetiveWord', 'toneShapeCode', 'qa',
            'AliasMapTABLE'
        ]
        self.hnsw = HNSW(cfg, is_rough=True)

    def test_predict(self):
        dbs = self.mongo.show_dbs()
        cols = self.mongo.show_collections()

        expect_output = self.hnsw.get_data_count()
        output = self.mongo.get_col_stats(
            'qa')['executionStats']['totalDocsExamined']

        self.assertEqual(output, expect_output)

        self.assertTrue(set(dbs).issuperset(set(self.db)))
        self.assertTrue(set(cols).issuperset(set(self.collections)))
