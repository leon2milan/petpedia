import unittest
from config import get_cfg
from qa.knowledge.entity_link import EntityLink


class EntitylinkTest(unittest.TestCase):
    def setUp(self):
        cfg = get_cfg()
        self.model_tester = EntityLink(cfg)
        self.text = [
            "狗乱吃东西怎么办", "边牧偶尔尿血怎么办", "猫咪经常拉肚子怎么办", "哈士奇拆家怎么办", "英短不吃东西怎么办？",
            "拉布拉多和金毛谁聪明", "折耳怀孕不吃东西怎么办？", "阿提桑诺曼底短腿犬", "阿尔卑斯达切斯勃拉克犬",
            "狗狗骨折了怎么办", '金毛一直咳嗽检查说是支气管肺炎及支气管扩张怎么治'
        ]

    def test_predict(self):
        expect_output = [('异食癖', 'SYMPTOMS'), ('血尿', 'SYMPTOMS'),
                         ('腹泻', 'SYMPTOMS'), ('西伯利亚哈士奇犬', 'DOG'),
                         ('厌食', 'SYMPTOMS'), ('金毛寻回猎犬', 'DOG'),
                         ('厌食', 'SYMPTOMS'), ('阿提桑诺曼底短腿犬', 'DOG'),
                         ('阿尔卑斯达切斯勃拉克犬', 'DOG'), ('骨折（犬）', 'DISEASE'),
                         ('支气管肺炎（犬）', 'DISEASE')]
        output = []
        for x in self.text:
            output.append(self.model_tester.entity_link(x))
        self.assertEquals(output, expect_output)
