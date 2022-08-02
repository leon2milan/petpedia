import unittest
from qa.tools.es import ES
from config import get_cfg
from qa.tools.mongo import Mongo


class ModelTest(unittest.TestCase):

    def setUp(self):
        cfg = get_cfg()
        self.mongo = Mongo(cfg, cfg.BASE.QA_COLLECTION)
        self.es = ES(cfg)
        self.index = ['qa_v1']

    def test_predict(self):
        expect_output = [
            self.mongo.get_col_stats('qa')['executionStats']
            ['totalDocsExamined']
        ]
        output = []
        for idx in self.index:
            output.append(self.es.es.count(index=idx)['count'])
        self.assertEqual(output, expect_output)