import unittest
from config import get_cfg
from qa.main import Search
from qa.tools import setup_logger
logger = setup_logger()

class SearchTest(unittest.TestCase):
    def setUp(self):
        cfg = get_cfg()
        self.model_tester = Search(cfg)
        self.text = [
            '狗狗容易感染什么疾病', '哈士奇老拆家怎么办', '犬瘟热', '狗发烧', '金毛', '拉布拉多不吃东西怎么办',
            '犬细小病毒的症状', '犬细小', '我和的', '阿提桑诺曼底短腿犬'
        ]

    def test_predict(self):
        expect_output = [('腊肠犬容易得的疾病', 0.92),
                         ('哈士奇拆家吗', 0.85),
                         ('犬瘟热怎么得的', 1.0),
                         ('狗发烧了怎么退烧', 0.97),
                         ('金毛參展標準', 1.0),
                         ('拉布拉多不吃东西怎么回事', 0.94),
                         ('犬细小病毒的症状', 1.0), ('犬细小传染狗吗', 0.98), [],
                         ('柯基犬怀孕吃什么', 0.0)]
        output = []
        for x in self.text:
            res = self.model_tester.search(x)
            if len(res) > 0:
                output.append([(x['doc']['question'], round(x['score'], 2))
                               for x in res][0])
            else:
                output.append([])
        logger.info(output)
        self.assertEquals(output, expect_output)
