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
        expect_output = [('防治狗狗容易生病的疾病及方法', 20.7),
                         ('哈士奇拆家吗', 21.1),
                         ('犬瘟热转阴', 21.35),
                         ('狗发烧的症状', 21.08),
                         ('金毛九色', 21.17),
                         ('拉布拉多不吃东西怎么回事', 21.6),
                         ('犬细小病毒的症状', 22.0), ('犬细小潜伏期', 21.14), [],
                         ('柯基犬怀孕吃什么', 19.18)]
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
