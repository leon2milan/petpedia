import unittest
from config import get_cfg
from qa.search.search_helper import SearchHelper


class SearchhelperTest(unittest.TestCase):
    def setUp(self):
        cfg = get_cfg()
        self.model_tester = SearchHelper(cfg)

    def test_hotfreq(self):
        expect_output = [
            '金毛犬怎么训练握手', '狗狗驱虫要注意什么', '小狗呕吐拉稀怎么办', '猫得了猫藓怎么治', '犬细小病毒的症状'
        ]
        output = [x['doc']['question'] for x in self.model_tester.hot_query()]
        self.assertEqual(output, expect_output)

