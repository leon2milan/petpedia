import unittest
from qa.tools.mongo import Mongo
from config import get_cfg


class ModelTest(unittest.TestCase):
    def setUp(self):
        cfg = get_cfg()
        self.mongo = Mongo(cfg, cfg.INVERTEDINDEX.DB_NAME)
        self.files = ['qa', 'sensetiveWord', 'toneShapeCode']

    def test_predict(self):
        expect_output = [208633, 41105, 7238]
        output = []
        for i in self.files:
            output.append(len(list(self.mongo.find(i, {}))))
        self.assertEquals(output, expect_output)