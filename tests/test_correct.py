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
        expect_output = [
            ((1, 2), '砂', 7.06),
            ((4, 8), '没有精神', 9.11),
            ((-1, -1), '', 0.0),
            ((0, 5), '猫咪', 15700),
            ((2, 5), '拉肚子', 21.96),
            ((4, 6), '拉稀', 12.63),
            ((4, 8), '尿血', 17.04),
            ((-1, -1), '', 0.0),
            ((4, 6), '感冒', 13.53),
            ((0, 2), '传染', 10.09),
            ((0, 2), '呕吐', 11.38),
            ((2, 3), '咳', 10.56),
            ((3, 4), '个', 12.39),
            ((-1, -1), '', 0.0),
            ((-1, -1), '', 0.0),
            ((0, 3), '犬瘟热', 1.0),
        ]
        output = []
        for x in self.text:
            e_pos, candidate, score = self.model_tester.correct(x)
            output.append((e_pos, candidate, round(score, 2)))
        self.assertEquals(output, expect_output)
