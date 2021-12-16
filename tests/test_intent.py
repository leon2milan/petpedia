import unittest
from config import get_cfg
from qa.intent.fast_text import Fasttext


class FasttextTest(unittest.TestCase):
    def setUp(self):
        cfg = get_cfg()
        self.model_tester = Fasttext(cfg, 'two_intent')
        self.text = [
            "拉布拉多不吃东西怎么办", "请问是否可以帮忙鉴别品种", "金毛犬如何鉴定", "发烧", "拉肚子", "感冒", '掉毛',
            '我和的', '阿提桑诺曼底短腿犬', '胰腺炎'
        ]

    def test_predict(self):
        expect_output = [
            'pet_qa', 'pet_qa', 'pet_qa', 'pet_qa', 'pet_qa', 'pet_qa',
            'pet_qa', 'chitchat', 'pet_qa', 'pet_qa'
        ]
        output = []
        for x in self.text:
            output.append(self.model_tester.predict(x)[0])
        self.assertEquals(output, expect_output)
