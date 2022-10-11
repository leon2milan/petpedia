import os
import re
from functools import reduce

import fasttext
import pandas as pd
from config import get_cfg
from qa.queryUnderstanding.preprocess.preprocess import nonsence_detect
from qa.queryUnderstanding.querySegmentation import Segmentation, Words
from qa.tools import flatten
from qa.tools.ahocorasick import Ahocorasick
from qa.tools.mongo import Mongo

INTENT_MAP = {
    '__label__2': "sensetive",
    '__lable__1': "pet_qa",
    "__lable__0": "chitchat"
}
__all__ = ['Fasttext']


# The class takes in a text, and returns the intent and the probability of the text
class Fasttext(object):
    __slot__ = [
        'cfg', 'mongo', 'seg', 'stopwords', 'specialize', 'classifier', 'ah', 'exclusive'
    ]

    def __init__(self, cfg, model=None):
        self.cfg = cfg
        self.mongo = Mongo(self.cfg, self.cfg.BASE.QA_COLLECTION)
        self.seg = Segmentation(self.cfg)
        self.stopwords = Words(self.cfg).get_stopwords
        self.specialize = Words(cfg).get_specializewords
        model = model if model and model is not None else 'intent'

        self.exclusive = ['发货', '不能', '失败', '没有', '能用']
        self.build_detector()
        self.build_sensetive_detector()
        if model is None or not os.path.exists(os.path.join(self.cfg.INTENT.MODEL_PATH, f'{model}.bin')):
            self.train(model)
        else:
            self.classifier = fasttext.load_model(os.path.join(self.cfg.INTENT.MODEL_PATH, f'{model}.bin'))

    def build_sensetive_detector(self):
        """
        It takes a list of words, and builds a tree structure that allows you to search for all of those
        words in a string in O(n) time
        """
        sw = Words(self.cfg).get_sensitivewords
        self.sen_detector = Ahocorasick()
        for word in sw:
            self.sen_detector.add_word(word)
        self.sen_detector.make()

    def sensetive_detect(self, query):
        """
        It takes a string as input, and returns a tuple of two elements: a boolean value and a list of
        strings
        
        :param query: the query string
        :return: A tuple of two elements. The first element is a boolean value indicating whether the
        query contains sensitive words. The second element is a list of sensitive words.
        """
        res = self.sen_detector.search_all(query)
        flag = False
        if len(res) > 0:
            flag = True
            res = [query[x[0]:x[1] + 1] for x in res]
        return flag, res

    def build_detector(self):
        """
        1. If the file `detector.txt` exists, read it into a list `qa`; otherwise, read the data from
        the database, and then do some preprocessing to get the list `qa`，
        2. Then to build a Ahocorasick
        """
        self.ah = Ahocorasick()
        if os.path.exists(os.path.join(self.cfg.INTENT.MODEL_PATH, 'detector.txt')):
            qa = [x.strip() for x in open(os.path.join(self.cfg.INTENT.MODEL_PATH, 'detector.txt')).readlines() if x.strip() not in self.exclusive]

        else:
            qa = self._extracted_from_build_detector_13()
        for word in qa:
            self.ah.add_word(word)
        self.ah.make()

    # TODO Rename this here and in `build_detector`
    def _extracted_from_build_detector_13(self):
        entity_word = flatten([[k, v] for k, v in reduce(lambda a, b: dict(a, **b), self.specialize.values()).items()])

        result = pd.DataFrame(list(self.mongo.find(self.cfg.BASE.QA_COLLECTION, {})))
        result['question_cut'] = result['question'].apply(lambda x: list(self.seg.cut(x, mode='pos', is_rough=True)))
        result['question_cut'] = result['question_cut'].apply(lambda x: list(zip(x[0], x[1])))
        result['question_cut'] = result['question_cut'].apply(lambda x: [i[0] for i in x if i[1] in ['n', 'nz', 'v', 'vn', 'nw']])
        result = flatten(result['question_cut'].apply(lambda x: [y for y in x if y not in self.stopwords]).tolist())
        result = [x for x in list(set(result)) if len(x) > 1]
        # result = []
        result += entity_word
        with open(os.path.join(self.cfg.INTENT.MODEL_PATH, 'detector.txt'), 'w') as f:
            for i in result:
                if i not in self.exclusive:
                    f.write(i + '\n')
        return result

    def load_data(self, to_file):
        """
        1. Load the data from the file, and then randomly select 1% of the data as the training data.
        2. The data is divided into positive and negative samples.
        3. The positive sample is the data in the mongo database, and the negative sample is the data in
        the file.
        4. The data is randomly shuffled and saved to the file
        
        :param to_file: the name of the file to be saved
        """
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
        """
        The function takes in a file name, checks if the file exists, if it does, it loads the data,
        then it trains the classifier, saves the model, and prints the result
        
        :param file_name: The name of the file to be trained
        """
        if '.' not in file_name:
            file_name = f'{file_name}.csv'
        file_path = os.path.join(self.cfg.INTENT.DATA_PATH, file_name)
        if os.path.exists(file_path):
            self.load_data(file_name)
        self.classifier = fasttext.train_supervised(file_path, label='__lable__', dim=200, epoch=20)

        self.classifier.save_model(os.path.join(self.cfg.INTENT.MODEL_PATH, file_name.replace('csv', 'bin')))

        result = self.classifier.test(file_path)
        print("result:", result)
        # print("P@1:", result.precision)  #准确率
        # print("R@2:", result.recall)  #召回率
        # print("Number of examples:", result.nexamples)  #预测错的例子

    def predict(self, text):
        """
        1. If the input text is empty, return "chitchat" and 1.0.
        2. If the input text is sensetive, return "sensetive" and 1.0.
        3. If the input text is a question, return "pet_qa" and 1.0.
        4. If the input text is nonsence, return "chitchat" and 1.0.
        5. If the input text is not empty, return the result of the classifier
        
        :param text: the text to be classified
        """

        if not text:
            return "chitchat", 1.0
        flag, sen = self.sensetive_detect(text)
        if flag:
            return 'sensetive', 1.0
        
        res = self.ah.search(text)
        if res:
            return "pet_qa", 1.0

        flag = nonsence_detect(text)
        if flag:
            return "chitchat", 1.0
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
        '我和的', '阿提桑诺曼底短腿犬', '胰腺炎', 'hello', '金毛相似品种', '习大大', '犬细小病毒的症状', '牙菌斑',
        u'\U0001F947', u':goldmedaille:', ':/::B', "猫", "狗", "我的订单没有发货", "商城优惠券不能用", "预约挂号失败",
        '€??x榐鹛)'
    ]

    for x in text:
        print(x, intent.predict(x))
