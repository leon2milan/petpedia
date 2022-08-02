import unittest
from config import get_cfg
from qa.contentUnderstanding import ContentUnderstanding


class CorrectTest(unittest.TestCase):
    def setUp(self):
        cfg = get_cfg()
        self.model_tester = ContentUnderstanding(cfg)
        self.text = ['狗狗容易感染什么疾病', '哈士奇老拆家怎么办', '犬瘟热', '狗发烧']

    def test_predict(self):
        expect_output = [{
            'SPECIES': 'DOG'
        }, {
            'SPECIES': 'DOG',
            'breed_name': '西伯利亚哈士奇犬'
        }, {
            'SPECIES': 'DOG',
            'disease_name': '犬瘟热病毒感染'
        }, {
            'SPECIES': 'DOG'
        }]
        output = []
        for x in self.text:
            output.append({
                k: v[0]
                for k, v in self.model_tester.understanding(x).items() if v
            })
        self.assertEqual(output, expect_output)
