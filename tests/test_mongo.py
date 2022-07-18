import unittest
from qa.tools.mongo import Mongo
from config import get_cfg


class ModelTest(unittest.TestCase):
    def setUp(self):
        cfg = get_cfg()
        self.mongo = Mongo(cfg, cfg.BASE.QA_COLLECTION)
        self.files = ['qa', 'sensetiveWord', 'toneShapeCode']

    def test_predict(self):
        expect_output = [207734, 41102, 7238]
        output = []
        for i in self.files:
            output.append(len(list(self.mongo.find(i, {}))))
        self.assertEqual(output, expect_output)