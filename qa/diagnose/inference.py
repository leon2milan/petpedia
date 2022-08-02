import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from config import get_cfg, ROOT
from qa.tools.mongo import Mongo
import copy
from qa.tools.utils import to_pinyin
import os


def softmax(x):
    f_x = x / np.sum(x)
    return f_x


class SelfDiagnose:

    def __init__(self, cfg):
        self.cfg = cfg
        self.mongo = Mongo(cfg, cfg.BASE.QA_COLLECTION)
        self.use_class_weight = False
        self.init()

    def init(self):
        relation = pd.read_excel(
            os.path.join(ROOT, 'data/diagose/sym-disease.xlsx'),
            engine='openpyxl',
            sheet_name='Sheet1')[['disease', 'symptom', 'weight']]
        tag = pd.DataFrame(list(self.mongo.find('diseaseReason', {})))
        tag.columns = [
            '_id', 'disease', 'breed', 'label', 'immune', 'sterilization',
            'age', 'sex', 'inner_deworm', 'outer_denorm', '_class'
        ]

        self.breed_map = {'猫': 0, '犬': 1, 'cat': 0, 'dog': 1}
        self.age_map = {
            '未知': 0,
            '幼年': 1,
            '中年': 2,
            '老年': 3,
            'unknown': 0,
            'youth': 1,
            'medium': 2,
            'old': 3,
            None: 0
        }
        self.sex_map = {
            '未知': 0,
            '雄性': 1,
            '雌性': 2,
            'unknown': 0,
            'boy': 1,
            'girl': 2,
            None: 0
        }
        self.jueyu_map = {
            '未知': 0,
            '已绝育': 1,
            '未绝育': 2,
            'unknown': 0,
            'yes': 1,
            'no': 2,
            None: 0
        }
        self.mainyi_map = {
            '未知': 0,
            '已免疫': 1,
            '未免疫': 2,
            'unknown': 0,
            'yes': 1,
            'no': 2,
            None: 0
        }
        self.inner_map = {
            '未知': 0,
            '已驱虫': 1,
            '未驱虫': 2,
            'unknown': 0,
            'yes': 1,
            'no': 2,
            None: 0
        }
        self.outter_map = {
            '未知': 0,
            '已驱虫': 1,
            '未驱虫': 2,
            'unknown': 0,
            'yes': 1,
            'no': 2,
            None: 0
        }

        tag['breed'] = tag['breed'].map(self.breed_map)
        tag['immune'] = tag['immune'].map(self.mainyi_map)
        tag['sterilization'] = tag['sterilization'].map(self.jueyu_map)
        tag['age'] = tag['age'].map(self.age_map)
        tag['sex'] = tag['sex'].map(self.sex_map)
        tag['inner_deworm'] = tag['inner_deworm'].map(self.inner_map)
        tag['outer_denorm'] = tag['outer_denorm'].map(self.outter_map)

        relation = relation.merge(tag, how='inner', on='disease')
        symptom = relation['symptom'].unique().tolist()
        symptom = sorted(list(set([x for x in symptom if x])),
                         key=to_pinyin,
                         reverse=False)
        max_sym = len(symptom)
        graph = copy.deepcopy(relation)
        graph['symptom'] = graph['symptom'] + ':' + graph['weight'].astype(str)
        graph = graph.groupby('disease', as_index=False).agg(
            {'symptom': lambda row: list(row)})
        graph['symptom'] = graph['symptom'].apply(lambda x: sorted(
            x,
            key=lambda y: (int(y.split(':')[-1]), y.split(':')[0]),
            reverse=True))

        self.sym_dict = {symptom[i]: i for i in range(len(symptom))}
        self.re_sym_dict = {v: k for k, v in self.sym_dict.items()}
        res = []
        for idx, row in graph.iterrows():
            sym = row['symptom']
            dis = row['disease']
            tmp = [0] * len(symptom)
            for i in symptom:
                for s in sym:
                    s, w = s.split(':')
                    tmp[self.sym_dict[s]] = str(w)
            res.append([dis] + [sym] + tmp)

        graph = pd.DataFrame(
            res,
            columns=['disease', 'symptom'] +
            ['symptom_' + str(i) for i in range(len(symptom))])
        for i in range(len(symptom)):
            graph['symptom_' + str(i)] = graph['symptom_' + str(i)].astype(int)

        test = copy.deepcopy(
            graph.merge(relation[[
                'disease', 'breed', 'immune', 'sterilization', 'age', 'sex',
                'inner_deworm', 'outer_denorm'
            ]].drop_duplicates(),
                        how='inner',
                        on='disease'))
        # test = test.drop('symptom', 1)
        test['symptom'] = test['symptom'].apply(
            lambda x: [y.split(':')[0] for y in x])
        disease = test['disease'].tolist()
        disease = sorted(list(set([x for x in disease if x])),
                         key=to_pinyin,
                         reverse=False)  # + ['Head']
        self.dis_dict = {disease[i]: i for i in range(len(disease))}
        self.re_dis_dict = {v: k for k, v in self.dis_dict.items()}
        self.init_weight()

        test['is_immune'] = test['immune'].apply(lambda x: 1
                                                 if x == 0 or x == 1 else 0)
        test['not_immune'] = test['immune'].apply(lambda x: 1
                                                  if x == 0 or x == 2 else 0)

        test['is_sterilization'] = test['sterilization'].apply(
            lambda x: 1 if x == 0 or x == 1 else 0)
        test['not_sterilization'] = test['sterilization'].apply(
            lambda x: 1 if x == 0 or x == 2 else 0)

        test['is_inner_deworm'] = test['inner_deworm'].apply(
            lambda x: 1 if x == 0 or x == 1 else 0)
        test['not_inner_deworm'] = test['inner_deworm'].apply(
            lambda x: 1 if x == 0 or x == 2 else 0)

        test['is_outer_denorm'] = test['outer_denorm'].apply(
            lambda x: 1 if x == 0 or x == 1 else 0)
        test['not_outer_denorm'] = test['outer_denorm'].apply(
            lambda x: 1 if x == 0 or x == 2 else 0)

        test['is_mal'] = test['sex'].apply(lambda x: 1
                                           if x == 0 or x == 2 else 0)
        test['is_female'] = test['sex'].apply(lambda x: 1
                                              if x == 0 or x == 1 else 0)

        test['label'] = test['disease'].map(self.dis_dict)
        train = test[[
            'label',
            'symptom_0',
            'symptom_1',
            'symptom_2',
            'symptom_3',
            'symptom_4',
            'symptom_5',
            'symptom_6',
            'symptom_7',
            'symptom_8',
            'symptom_9',
            'symptom_10',
            'symptom_11',
            'symptom_12',
            'symptom_13',
            'symptom_14',
            'symptom_15',
            'symptom_16',
            'symptom_17',
            'symptom_18',
            'symptom_19',
            'symptom_20',
            'symptom_21',
            'symptom_22',
            'symptom_23',
            'symptom_24',
            'symptom_25',
            'symptom_26',
            'symptom_27',
            'symptom_28',
            'symptom_29',
            'symptom_30',
            'symptom_31',
            'symptom_32',
            'symptom_33',
            'symptom_34',
            'symptom_35',
            'symptom_36',
            'symptom_37',
            'symptom_38',
            'symptom_39',
            'symptom_40',
            'symptom_41',
            'symptom_42',
            'breed',
        ]]
        self.columns = [x for x in train.columns if x != 'label']
        for col in self.columns:
            train[col] = train[col].apply(lambda x: 1 if x > 0 else 0)

        x_train = train.drop('label', axis=1)  # 获得训练集的x
        y_train = train['label']  # 获取训练集的y
        x_test = train.drop('label', axis=1)  # 获取测试集的x
        y_test = train['label']  # 获取测试集的y
        self.model = self.build(x_train, y_train, self.use_class_weight)
        self.init_filter(tag, relation)

    def init_filter(self, tag, relation):
        self.breed_filter = dict(zip(tag['disease'], tag['breed']))
        self.sym_dis = {
            x['symptom']: x['disease']
            for x in relation[['symptom', 'disease']].groupby(
                'symptom', as_index=False).agg({
                    'disease': list
                }).to_dict(orient='record')
        }

        self.is_immune = tag[tag['immune'] == 1]['disease'].tolist()
        self.not_immune = tag[tag['immune'] == 2]['disease'].tolist()

        self.is_sterilization = tag[tag['sterilization'] ==
                                    1]['disease'].tolist()
        self.not_sterilization = tag[tag['sterilization'] ==
                                     2]['disease'].tolist()

        self.is_female = tag[tag['sex'] == 2]['disease'].tolist()
        self.is_male = tag[tag['sex'] == 1]['disease'].tolist()

        self.is_inner_deworm = tag[tag['inner_deworm'] ==
                                   1]['disease'].tolist()
        self.not_inner_deworm = tag[tag['inner_deworm'] ==
                                    2]['disease'].tolist()

        self.is_outer_denorm = tag[tag['outer_denorm'] ==
                                   1]['disease'].tolist()
        self.not_outer_denorm = tag[tag['outer_denorm'] ==
                                    2]['disease'].tolist()

        self.is_young = tag[tag['age'] == 1]['disease'].tolist()
        self.is_medium = tag[tag['age'] == 2]['disease'].tolist()
        self.is_old = tag[tag['age'] == 3]['disease'].tolist()

        self.weight_dict = {}
        for idx, row in relation.iterrows():
            if row['symptom'] not in self.weight_dict:
                self.weight_dict[row['symptom']] = {}
            self.weight_dict[row['symptom']][row['disease']] = row['weight']

    def init_weight(self):
        count = pd.read_excel(
            os.path.join(ROOT, 'data/diagose/disease_count.xlsx'),
            engine='openpyxl',
            sheet_name='Sheet1')[['disease', 'count']].fillna(1.0)
        count['label'] = count['disease'].map(self.dis_dict).astype(int)
        count['count'] = count['count'] / count['count'].sum()
        self.class_weight = {
            x['label']: x['count']
            for x in count[['label', 'count']].to_dict(orient='record')
        }

    def build(self, x_train, y_train, use_class_weight=False):
        if use_class_weight:
            rf_model = RandomForestClassifier(random_state=18,
                                              n_estimators=100,
                                              class_weight=self.class_weight)
        else:
            rf_model = RandomForestClassifier(random_state=18,
                                              n_estimators=100)
        rf_model.fit(x_train, y_train)
        print(
            f"Accuracy on train data by Random Forest Classifier: {accuracy_score(y_train, rf_model.predict(x_train))*100}"
        )
        return rf_model

    def filtering(self,
                  proba,
                  breed,
                  symptoms,
                  immune=None,
                  sterilization=None,
                  age=None,
                  sex=None,
                  inner_deworm=None,
                  outer_denorm=None):
        #     for d in [k for k, v in breed_filter.items() if v == 1 - breed]:
        #         proba[dis_dict[d]] = 0.0
        dis_candidate = [d for x in symptoms for d in self.sym_dis[x]]
        for d, _ in self.dis_dict.items():
            if d not in dis_candidate:
                proba[self.dis_dict[d]] = 0.0
            else:
                if self.breed_filter[d] != breed:
                    proba[self.dis_dict[d]] = 0.0

        if immune is not None:
            if immune == 1:  # 已免疫
                for d in self.not_immune:
                    proba[self.dis_dict[d]] = 0.0
            elif immune == 2:  # 未免疫
                for d in self.is_immune:
                    proba[self.dis_dict[d]] = 0.0

        if sterilization is not None:
            if sterilization == 1:  # 已绝育
                for d in self.not_sterilization:
                    proba[self.dis_dict[d]] = 0.0
            elif sterilization == 2:  # 未绝育
                for d in self.is_sterilization:
                    proba[self.dis_dict[d]] = 0.0

        if age is not None:
            if age == 1:  # 幼年
                for d in self.is_medium + self.is_old:
                    proba[self.dis_dict[d]] = 0.0
            elif age == 2:  # 中年
                for d in self.is_young + self.is_old:
                    proba[self.dis_dict[d]] = 0.0
            elif age == 3:  # 老年
                for d in self.is_young + self.is_medium:
                    proba[self.dis_dict[d]] = 0.0

        if sex is not None:
            if sex == 1:  # 雄性
                for d in self.is_female:
                    proba[self.dis_dict[d]] = 0.0
            elif sex == 2:  # 雌性
                for d in self.is_male:
                    proba[self.dis_dict[d]] = 0.0

        if inner_deworm is not None:
            if inner_deworm == 1:  # 已体内驱虫
                for d in self.not_inner_deworm:
                    proba[self.dis_dict[d]] = 0.0
            elif inner_deworm == 2:  # 未体内驱虫
                for d in self.is_inner_deworm:
                    proba[self.dis_dict[d]] = 0.0

        if outer_denorm is not None:
            if outer_denorm == 1:  # 已体外驱虫
                for d in self.not_outer_denorm:
                    proba[self.dis_dict[d]] = 0.0
            elif outer_denorm == 2:  # 未体外驱虫
                for d in self.is_outer_denorm:
                    proba[self.dis_dict[d]] = 0.0
        return proba

    def reweighting(self, proba, sym):
        dis = self.weight_dict[sym]
        for d, w in dis.items():
            proba[self.dis_dict[d]] *= w
        return proba

    def diagnose(self, request, topk=5):
        inputs = dict(zip(self.columns, [0] * (len(self.columns))))
        for sym in request['symptom']:
            inputs['symptom_' + str(self.sym_dict[sym])] = 1
        inputs['breed'] = self.breed_map[request['pet']]

        proba = self.model.predict_proba(pd.DataFrame([inputs]))[0]

        inputs['age'] = self.age_map[request.get('age', '未知')]
        inputs['immune'] = self.mainyi_map[request.get('is_mianyi', '未知')]
        inputs['sterilization'] = self.jueyu_map[request.get('is_jueyu', '未知')]
        inputs['sex'] = self.sex_map[request.get('sex', '未知')]
        inputs['inner_deworm'] = self.inner_map[request.get(
            'inner_deworm', '未知')]
        inputs['outer_denorm'] = self.outter_map[request.get(
            'outter_deworm', '未知')]

        # print([(self.re_dis_dict[x], proba[x]) for x in (-proba).argsort()[:topk].tolist()])

        proba = self.filtering(proba, inputs['breed'], request['symptom'],
                               inputs['immune'], inputs['sterilization'],
                               inputs['age'], inputs['sex'],
                               inputs['inner_deworm'], inputs['outer_denorm'])
        for sym in request['symptom']:
            proba = self.reweighting(proba, sym)

        proba = softmax(proba)

        return [{
            'disease': self.re_dis_dict[x],
            'probility': proba[x]
        } for x in (-proba).argsort()[:topk].tolist()]


if __name__ == "__main__":
    # request = {'pet': '猫', 'symptom': ['呕吐']}
    # request = {'pet': '犬', 'symptom': ['胀气']}
    # request = {'pet': '犬', 'symptom': ['吐血']}
    # request = {'pet': '犬', 'symptom': ['体重下降']}
    # request = {'pet': '犬', 'symptom': ['咳嗽']}
    # request = {'pet': '犬', 'symptom': ['便血', '腹泻']}
    # request = {'pet': '犬', 'symptom': ['阴门分泌物']}
    # request = {'pet': '犬', 'symptom': ['胀气', '阴门分泌物']}
    # request = {
    #     'symptom': ['口臭'],
    #     'pet': '犬',
    #     'is_mianyi': None,
    #     'inner_deworm': None,
    #     'outter_deworm': None,
    #     'age': '未知',
    #     'sex': '未知',
    #     'is_jueyu': None
    # }
    request = {
        'symptom': ['阴门分泌物'],
        'pet': '猫',
        'is_mianyi': None,
        'inner_deworm': None,
        'outter_deworm': None,
        'age': '未知',
        'sex': '雌性',
        'is_jueyu': None
    }
    # request = {'pet': '犬', 'symptom': ['包皮分泌物']}
    # request = {'pet': '猫', 'symptom': ['多食']}
    cfg = get_cfg()
    sd = SelfDiagnose(cfg)
    print(sd.diagnose(request, 5))
    