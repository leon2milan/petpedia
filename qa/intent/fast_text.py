from config import get_cfg
from qa.tools.mongo import Mongo
import pandas as pd
import fasttext
import os
from qa.queryUnderstanding.querySegmentation import Segmentation, Words

INTENT_MAP = {'__lable__1': "pet_qa", "__lable__0": "chitchat"}


class Fasttest(object):
    def __init__(self, cfg, model=None):
        self.cfg = cfg
        self.mongo = Mongo(self.cfg, self.cfg.INVERTEDINDEX.DB_NAME)
        self.seg = Segmentation(self.cfg)
        self.stopwords = Words(self.cfg).get_stopwords
        model = model if model and model is not None else 'intent'
        if model is None or not os.path.exists(
                os.path.join(self.cfg.INTENT.MODEL_PATH,
                             '{}.bin'.format(model))):
            self.train(model)
        else:
            self.classifier = fasttext.load_model(
                os.path.join(self.cfg.INTENT.MODEL_PATH,
                             '{}.bin'.format(model)))

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

        data = pd.concat([positive['content'], negtive['content']])
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

    def predict(self, text_list):
        text_list = [" ".join(x) for x in self.seg.cut(text_list)]
        lable, probs = self.classifier.predict(text_list)
        return INTENT_MAP[lable[0][0]], probs[0].tolist()[0]


if __name__ == '__main__':
    cfg = get_cfg()
    intent = Fasttest(cfg, 'two_intent')

    text = ["拉布拉多不吃东西怎么办", "金毛犬如何鉴定"]

    print(intent.predict(text))