import os
import re
from functools import reduce

import fasttext
import pandas as pd
from config import get_cfg
from qa.queryUnderstanding.querySegmentation import Segmentation, Words
from qa.tools import flatten
from qa.tools.ahocorasick import Ahocorasick
from qa.tools.mongo import Mongo

INTENT_MAP = {'__lable__1': "pet_qa", "__lable__0": "chitchat"}
__all__ = ['Fasttext']


class Fasttext(object):
    __slot__ = [
        'cfg', 'mongo', 'seg', 'stopwords', 'specialize', 'classifier', 'ah'
    ]

    def __init__(self, cfg, model=None):
        self.cfg = cfg
        self.mongo = Mongo(self.cfg, self.cfg.BASE.QA_COLLECTION)
        self.seg = Segmentation(self.cfg)
        self.stopwords = Words(self.cfg).get_stopwords
        self.specialize = Words(cfg).get_specializewords
        model = model if model and model is not None else 'intent'
        self.build_detector()
        if model is None or not os.path.exists(
                os.path.join(self.cfg.INTENT.MODEL_PATH,
                             '{}.bin'.format(model))):
            self.train(model)
        else:
            self.classifier = fasttext.load_model(
                os.path.join(self.cfg.INTENT.MODEL_PATH,
                             '{}.bin'.format(model)))

    def build_detector(self):
        self.ah = Ahocorasick()
        if os.path.exists(
                os.path.join(self.cfg.INTENT.MODEL_PATH, 'detector.txt')):
            qa = [
                x.strip() for x in open(
                    os.path.join(self.cfg.INTENT.MODEL_PATH,
                                 'detector.txt')).readlines()
            ]

        else:
            entity_word = flatten([[k, v] for k, v in reduce(
                lambda a, b: dict(a, **b), self.specialize.values()).items()])
            qa = pd.DataFrame(
                list(self.mongo.find(self.cfg.BASE.QA_COLLECTION, {})))
            qa['question_cut'] = qa['question'].progress_apply(
                lambda x: self.seg.cut(x, mode='pos', is_rough=True))
            qa['question_cut'] = qa['question_cut'].apply(
                lambda x: list(zip(x[0], x[1])))
            qa['question_cut'] = qa['question_cut'].apply(
                lambda x:
                [i[0] for i in x if i[1] in ['n', 'nz', 'v', 'vn', 'nw']])

            qa = flatten(qa['question_cut'].apply(
                lambda x: [y for y in x if y not in self.stopwords]).tolist())
            qa = [x for x in list(set(qa)) if len(x) > 1]
            qa = qa + entity_word
            with open(os.path.join(self.cfg.INTENT.MODEL_PATH, 'detector.txt'),
                      'w') as f:
                for i in qa:
                    f.write(i + '\n')

        for word in qa:
            self.ah.add_word(word)
        self.ah.make()

    def load_data(self, to_file):
        base_path = os.path.join(self.cfg.BASE.DATA_PATH,
                                 'intent/clean_chat_corpus')
        intent_file = os.listdir(base_path)
        data = []
        for i_file in intent_file:
            df = pd.read_csv(os.path.join(base_path, i_file),
                             sep='\t',
                             names=['query', 'answer'],
                             header=None,
                             error_bad_lines=False)
            df['source'] = i_file.split('.')[0]
            data.append(df.sample(frac=0.01))
        data = pd.concat(data).reset_index(drop=True)

        negtive = pd.DataFrame(data.dropna()['query'].tolist() +
                               data.dropna()['answer'].tolist(),
                               columns=['content'])
        negtive['content'] = negtive['content'].apply(
            lambda x:
            [y for y in self.seg.cut(x) if y not in self.stopwords if y])
        negtive['len'] = negtive['content'].apply(len)
        negtive = negtive[negtive['len'] > 1]
        negtive['content'] = negtive['content'].apply(
            lambda x: " ".join(x) + "\t" + "__lable__" + str(0))

        positive = pd.DataFrame(
            list(self.mongo.find(self.cfg.BASE.QA_COLLECTION,
                                 {}))).dropna().sample(n=100000)
        positive['content'] = positive['question_fine_cut'].apply(
            lambda x: " ".join(x) + "\t" + "__lable__" + str(1))

        merge_all = reduce(lambda a, b: dict(a, **b), self.specialize.values())
        entity = pd.DataFrame(
            [re.sub(r'\（.*\）', '', x) for x in merge_all.keys() if x] +
            [x for y in merge_all.values() for x in y if x],
            columns=['entity']).drop_duplicates().reset_index(drop=True)
        entity['entity'] = entity['entity'].apply(lambda x: self.seg.cut(x))
        entity['content'] = entity['entity'].apply(
            lambda x: " ".join(x) + "\t" + "__lable__" + str(1))

        data = pd.concat(
            [positive['content'], negtive['content'], entity['content']])
        data.sample(frac=1).to_csv(os.path.join(self.cfg.INTENT.DATA_PATH,
                                                to_file),
                                   index=False,
                                   header=None)

    def train(self, file_name):
        if '.' not in file_name:
            file_name = '{}.csv'.format(file_name)
        file_path = os.path.join(self.cfg.INTENT.DATA_PATH, file_name)
        if os.path.exists(file_path):
            self.load_data(file_name)
        self.classifier = fasttext.train_supervised(file_path,
                                                    label='__lable__',
                                                    dim=200,
                                                    epoch=20)
        self.classifier.save_model(
            os.path.join(self.cfg.INTENT.MODEL_PATH,
                         file_name.replace('csv', 'bin')))
        result = self.classifier.test(file_path)
        print("result:", result)
        # print("P@1:", result.precision)  #准确率
        # print("R@2:", result.recall)  #召回率
        # print("Number of examples:", result.nexamples)  #预测错的例子

    def predict(self, text):
        res = self.ah.search(text)
        if res:
            return "pet_qa", 1.0
        text = " ".join(
            [x for x in self.seg.cut(text) if x not in self.stopwords])
        if not text:
            return "chitchat", 1.0
        lable, probs = self.classifier.predict(text)
        return INTENT_MAP[lable[0]], probs[0]


if __name__ == '__main__':
    cfg = get_cfg()
    intent = Fasttext(cfg, 'two_intent')

    text = [
        "拉布拉多不吃东西怎么办", "请问是否可以帮忙鉴别品种", "金毛犬如何鉴定", "发烧", "拉肚子", "感冒", '掉毛',
        '我和的', '阿提桑诺曼底短腿犬', '胰腺炎', 'hello', '金毛相似品种'
    ]

    for x in text:
        print(x, intent.predict(x))
