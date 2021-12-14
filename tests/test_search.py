import unittest
from config import get_cfg
from qa.main import Search


class SearchTest(unittest.TestCase):
    def setUp(self):
        cfg = get_cfg()
        self.model_tester = Search(cfg)
        self.text = [
            '狗狗容易感染什么疾病', '哈士奇老拆家怎么办', '犬瘟热', '狗发烧', '金毛', '拉布拉多不吃东西怎么办',
            '犬细小病毒的症状', '犬细小', '我和的', '阿提桑诺曼底短腿犬'
        ]

    def test_predict(self):
        expect_output = [('哈士奇容易得哪些疾病', 0.55),
                         ('哈士奇拆家怎么办？', 0.87),
                         ('犬瘟热转阴', 0.88),
                         ('狗反复发烧', 0.8),
                         ('金毛叫什么', 0.63),
                         ('拉布拉多不吃东西怎么办', 1.0),
                         ('犬细小病毒的症状', 1.0), ('犬细小潜伏期', 0.82), [],
                         ('芬兰波美拉尼亚丝毛狗怎么美容 芬兰狐狸犬美容方法', 0.27)]
        output = []
        for x in self.text:
            res = self.model_tester.search(x)
            if len(res) > 0:
                output.append([(x['doc']['question'], round(x['score'], 2))
                               for x in res][0])
            else:
                output.append([])
        self.assertEquals(output, expect_output)
