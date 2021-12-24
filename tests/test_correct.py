import unittest
from config import get_cfg
from qa.queryUnderstanding.queryReformat.queryCorrection.correct import SpellCorrection


class CorrectTest(unittest.TestCase):
    def setUp(self):
        cfg = get_cfg()
        self.model_tester = SpellCorrection(cfg)
        self.text = [
            "猫沙盆",  # 错字
            "我家猫猫精神没有",  # 乱序
            "狗狗发了",  # 少字
            "maomi",  # 拼音
            "猫咪啦肚子怎么办",
            "我家狗拉啦稀了",  # 多字
            "狗狗偶尔尿xie怎么办",
            "狗老抽出怎么办",
            '我家猫猫感帽了',
            '传然给我',
            '呕土不止',
            '一直咳数',
            '我想找哥医生',
            "哈士奇老拆家怎么办",
            "狗狗发烧怎么办",
            "犬瘟热"
        ]

    def test_predict(self):
        expect_output = []
        output = []
        for x in self.text:
            e_pos, candidate, score = self.model_tester.correct(x)
            output.append((e_pos, candidate, round(score, 2)))
        self.assertEqual([], expect_output)
