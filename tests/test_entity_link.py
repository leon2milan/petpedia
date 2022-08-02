import unittest
from config import get_cfg
from qa.knowledge.entity_link import EntityLink


class EntitylinkTest(unittest.TestCase):
    def setUp(self):
        cfg = get_cfg()
        self.model_tester = EntityLink(cfg)
        self.text = [
            "狗乱吃东西怎么办", "边牧偶尔尿血怎么办", "猫咪经常拉肚子怎么办", "哈士奇拆家怎么办", "英短不吃东西怎么办？",
            "折耳怀孕不吃东西怎么办？", "阿提桑诺曼底短腿犬", "阿尔卑斯达切斯勃拉克犬",
        ]

    def test_predict(self):
        expect_output = [('异食癖', 'SYMPTOMS'), 
                         ('血尿', 'SYMPTOMS'), ('边境牧羊犬', 'DOG'),
                         ('腹泻', 'SYMPTOMS'), 
                         ('西伯利亚哈士奇犬', 'DOG'),
                         ('厌食', 'SYMPTOMS'), ('英国短毛猫', 'CAT'), 
                         ('厌食', 'SYMPTOMS'), ('苏格兰折耳猫', 'CAT'), 
                         ('阿提桑诺曼底短腿犬', 'DOG'),
                         ('阿尔卑斯达切斯勃拉克犬', 'DOG'), ]
        output = []
        for x in self.text:
            output.extend(self.model_tester.entity_link(x))
        self.assertEqual(output, expect_output)
