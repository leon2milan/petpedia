import unittest
from qa.tools.es import ES
from config import get_cfg


class ModelTest(unittest.TestCase):
    def setUp(self):
        cfg = get_cfg()
        self.es = ES(cfg)
        self.index = ['qa_v1']

    def test_predict(self):
        expect_output = [207734]
        output = []
        for idx in self.index:
            output.append(self.es.es.count(index=idx)['count'])
        self.assertEqual(output, expect_output)