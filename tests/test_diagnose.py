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
        expect_output = [['猫蛋白丢失性肾病', '猫肾上腺皮质机能亢进', '猫蛋白丢失性肠病', '猫糖尿病', '猫肠道寄生虫病'],
                         ['猫脑膜炎', '猫脑水肿', '猫肠梗阻', '猫中枢神经系统肿瘤', '猫会厌肿瘤'],
                         ['犬阴道炎', '犬子宫积液', '犬子宫内膜增生', '犬子宫蓄脓', '犬阴道肿瘤'],
                         ['猫胆囊破裂', '猫胆结石', '猫肝硬化', '猫胆管肿瘤', '猫溶血性贫血']]
        output = []
        for x in self.text:
            output.append(
                [m['disease'] for m in self.model_tester.diagnose(x)])
        self.assertEqual(output, expect_output)
