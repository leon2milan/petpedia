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
            ((1, 2), '砂', 7.060676892598471),
            ((4, 8), '没有精神', 9.11111111111111),
            ((-1, -1), '', 0.0),
            ((0, 5), '猫咪', 15700),
            ((2, 5), '拉肚子', 21.957357109367074),
            ((4, 6), '拉稀', 12.629406520298549),
            ((4, 8), '尿血', 17.036372775123233),
            ((-1, -1), '', 0.0),
            ((-1, -1), '', 0.0),
            ((0, 2), '传染', 10.092335619245256),
            ((0, 2), '呕吐', 11.383795314364964),
            ((2, 4), '咳嗽', 5.9749872745611725),
            ((3, 4), '个', 12.393067042032879),
            ((-1, -1), '', 0.0),
            ((-1, -1), '', 0.0),
            ((0, 3), '犬瘟热', 1.0),
        ]
        output = []
        for x in self.text:
            output.append(self.model_tester.correct(x))
        self.assertEquals(output, expect_output)
