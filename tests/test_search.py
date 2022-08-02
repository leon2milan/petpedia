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
        expect_output = [
            '狗狗打了疫苗就不会感染疾病了吗', '哈士奇拆家吗', '犬瘟热转阴', '狗肠炎发烧吗', '金毛怎么样',
            '拉布拉多呕吐不吃东西', '犬细小病毒症状', '犬细小症状', [], '萨摩耶犬吃什么'
        ]
        output = []
        for x in self.text:
            res = self.model_tester.search(x)
            if len(res) > 0:
                output.append([(x['doc']['question']) for x in res][0])
            else:
                output.append([])
        logger.info(output)
        self.assertEqual(output, expect_output)
