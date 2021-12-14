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
        expect_output = [('哈士奇容易得哪些疾病', 0.5545353293418884),
                         ('哈士奇拆家怎么办？', 0.8659767508506775),
                         ('犬瘟热转阴', 0.8832299709320068),
                         ('狗反复发烧', 0.797723114490509),
                         ('金毛叫什么', 0.6282012462615967),
                         ('拉布拉多不吃东西怎么办', 0.9999999403953552),
                         ('犬细小病毒的症状', 1.0), ('犬细小潜伏期', 0.8209661245346069), [],
                         ('芬兰波美拉尼亚丝毛狗怎么美容 芬兰狐狸犬美容方法', 0.2720837891101837)]
        output = []
        for x in self.text:
            res = self.model_tester.search(x)
            if len(res) > 0:
                output.append([(x['doc']['question'], x['score'])
                               for x in res][0])
            else:
                output.append([])
        self.assertEquals(output, expect_output)
