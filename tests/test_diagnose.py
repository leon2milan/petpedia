import unittest
from config import get_cfg
from qa.diagnose.inference import SelfDiagnose


class EntitylinkTest(unittest.TestCase):

    def setUp(self):
        cfg = get_cfg()
        self.model_tester = SelfDiagnose(cfg)
        self.text = [{
            'pet': '猫',
            'symptom': ['多食']
        }, {
            'pet': '猫',
            'symptom': ['呕吐']
        }, {
            'pet': '犬',
            'symptom': ['阴门分泌物']
        }, {
            'pet': '猫',
            'symptom': ['黄疸']
        }]

    def test_predict(self):
        expect_output = [['猫糖尿病', '猫肾上腺皮质机能亢进', '猫甲状腺机能亢进', '猫胰外分泌不足'],
                         ['猫中耳炎', '猫胃内异物', '猫甲状腺机能亢进', '猫鼻腔肿瘤'], ['犬子宫蓄脓'],
                         ['猫脂肪肝']]
        output = []
        for x in self.text:
            output.append(
                [m['disease'] for m in self.model_tester.diagnose(x)])
        self.assertEqual(output, expect_output)
