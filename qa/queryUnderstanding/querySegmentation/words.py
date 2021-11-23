"""
获取敏感词       get_sensitive_words
获取重要词       get_important_words
获取近义词       get_synonym_words
获取停用词
检测敏感词       sensitive_words_find
检测重要词       important_words_find
同义词替换       synonym_replace
停用词去除
dfa查找         dfa
"""
from qa.tools import Singleton, setup_logger, flatten
from qa.tools.mongo import Mongo
import os
import pandas as pd
from functools import reduce

logger = setup_logger()

@Singleton
class Words():
    """
    获取各种词库
    """
    _instance = None
    _first_init = True

    def __init__(self, cfg):
        self.cfg = cfg

        self.mongo = Mongo(cfg, cfg.INVERTEDINDEX.DB_NAME)

        if self._first_init:  # 默认True 第一次初始化置为False
            self._first_init = False
            self.stop_words = None
            self.sensitive_words = None  # 敏感词
            self.specialize_words = None  #  专有词
            self.synonym_words = None  # 同义词
            self.same_pinyin = None  # 同音
            self.same_stroke = None  # 同型
            self.init()

    @property
    def get_stopwords(self):
        return self.stop_words

    @property
    def get_sensitivewords(self):
        return self.sensitive_words

    @property
    def get_specializewords(self):
        return self.specialize_words

    @property
    def get_samestroke(self):
        return self.same_stroke

    @property
    def get_samepinyin(self):
        return self.same_pinyin

    def init(self):
        """相当于run() 同义词先跑"""
        self.get_synonym_words(
        )  # get_synonym_words 必须在 get_stop_words 之前, 对停用词采用同义词替换
        self.get_stop_words()
        self.get_specialize_words()
        self.get_sensitive_words()
        self.get_pinyin()
        self.get_stroke()
        return True

    def reload(self):
        return self.init()

    def get_synonym_words(self):
        pass

    def get_sensitive_words(self):
        """从mysql中获取敏感词表
        :param mysql: obj
        :return: dict {subsystem1:[word1,word2,..], subsystem2:[word11,word12...],...}
        """
        self.sensitive_words = flatten([
            x.strip() for y in os.listdir(self.cfg.DICTIONARY.SENSITIVE_PATH)
            for x in open(os.path.join(self.cfg.DICTIONARY.SENSITIVE_PATH,
                                       y)).readlines()
        ])

    def get_stop_words(self):
        self.stop_words = [
            x.strip()
            for x in open(self.cfg.DICTIONARY.STOP_WORDS).readlines()
        ]

    @staticmethod
    def normalize(dic):
        return {
            k.strip(): [x.strip() for x in v.split('|') if x]
            for k, v in dic.items()
        }

    def get_specialize_words(self):
        cat = pd.read_csv(
            self.cfg.BASE.CAT_DATA)[['chinese_name',
                                     'chinese_alias']].fillna('')
        dog = pd.read_csv(
            self.cfg.BASE.DOG_DATA)[['chinese_name',
                                     'chinese_alias']].fillna('')
        sym = pd.read_csv(
            self.cfg.BASE.SYMPTOM_DATA).dropna(how='all').fillna('')
        ill = {
            x.strip(): x.strip()
            for x in open(self.cfg.BASE.DISEASE_DATA).readlines()
        }
        cat = dict(zip(cat['chinese_name'].values,
                       cat['chinese_alias'].values))
        dog = dict(zip(dog['chinese_name'].values,
                       dog['chinese_alias'].values))
        sym = dict(zip(sym['symtom'].values, sym['alias'].values))
        self.specialize_words = {
            '猫': Words.normalize.__func__(cat),
            '犬': Words.normalize.__func__(dog),
            '症状': Words.normalize.__func__(sym),
            '疾病': Words.normalize.__func__(ill)
        }

        new_word_path = os.path.join(self.cfg.BASE.DATA_PATH,
                                     'dictionary/segmentation/new_word.csv')
        if os.path.exists(new_word_path):
            new_word = [x.strip() for x in open(new_word_path).readlines()]

        merge_all = reduce(lambda a, b: dict(a, **b),
                           self.specialize_words.values())
        tmp = list(
            set(
                list(merge_all.keys()) +
                [y for x in merge_all.values() for y in x if y]))

        custom_word = sorted([
            x.strip()
            for x in open(self.cfg.DICTIONARY.CUSTOM_WORDS).readlines()
        ])
        tmp = sorted(list(set(tmp + new_word + custom_word)))

        if tmp != custom_word:
            with open(self.cfg.DICTIONARY.CUSTOM_WORDS, 'w') as f:
                for word in tmp:
                    f.write(word + '\n')

    def get_pinyin(self):
        self.same_pinyin = {
            line.strip().split('\t')[0]:
            list("".join(line.strip().split('\t')[1:]))
            for line in open(
                self.cfg.DICTIONARY.SAME_PINYIN_PATH).readlines()[1:]
        }

    def get_stroke(self):
        self.same_stroke = {
            line.strip().split('\t')[0]:
            list("".join(line.strip().split('\t')[1:]))
            for line in open(
                self.cfg.DICTIONARY.SAME_STROKE_PATH).readlines()[1:]
        }

    def words_reload(self):
        """ 通过信号控制 sensitive_words important_words synonym_words random_resp 的更新删除
        :param mysql:
        :return: boolean
        """
        return True

    def synonym_replace(self, question, tables):
        """《同义词》替换
        :param question:
        :param tables: list 同义词列表
        :return: msg
        """
        raw = question
        # 方法1
        # pass_words = []
        # for table in tables:
        #     if not table:
        #         continue
        #     for replace_list, root_word in table:  # ([a, b, c], d)  d类似词根, a, b, c 为别名
        #         for w in replace_list:
        #             if w in question and w != root_word and w not in pass_words:
        #                 question = question.replace(w, root_word)
        #                 pass_words.append(root_word)
        # return question
        # 方法2
        for table in tables:
            if not table:
                continue
            for word_list in table:  # (a,b,c,d) 同义词列表, d最长
                holder = "!"
                root, t = word_list  # root:词根   t: 排好顺序的同义词列表
                for word in sorted(t, reverse=False):  # 优先替换长度长的
                    if root not in question:  # add by ian，12-13
                        question = question.replace(word, holder)
                question = question.replace(holder, root)

        if raw != question:
            logger.debug("同义词　{} -> {}".format(raw, question))
        return question

    def sensitive_words_find(self, question, tables):
        """ 《敏感词》检测 找到一个就返回
        :param question: str
        :param tables: dict
        :return: str
        """
        for table in tables:
            if not table:
                continue
            for word in table:
                if word in question:
                    # 防止曹操被过滤
                    if len(word) == 1:
                        if word == question:
                            return word
                    else:
                        return word

    def important_words_find(self, question, tables):
        """ 检查通用性《重要词》, 子系统自己的通用词 ,返回尽可能多的重要词
        :param question:
        :param tables:
        :return: []
        """
        ret = []
        for table in tables:
            if not table:
                continue
            for word in table:
                if word in question:
                    ret.append(word)
        return ret

    def stop_words_delete(self, question, tables):
        """
        :param question:
        :param tables:
        :return:
        """
        # if len(question) <= 3:  # 太短的语句不做停用词删除处理
        #     return question
        if not question:
            return ''
        question = question.translate(trantab)
        raw = question
        for table in tables:
            if not table:
                continue
            for word in table:
                if word in question:
                    # logger.debug("question={} 去除停用词:{}".format(question, word))
                    question = question.replace(word, "")
        if question != raw:
            logger.debug("去除停用词 {} --> {}".format(raw, question))
        return question or raw

    def rebuild(self, question):
        tables2 = self.synonym_words
        tables3 = self.stop_words
        # tables3 = [self.stop_words.get(comm), self.stop_words.get(subsystem_id)]
        # 同义词替换
        temp = self.synonym_replace(question, tables2)

        # 去除停用词
        temp = self.stop_words_delete(temp, tables3)

        # 同义词替换
        # tables2 = [self.synonym_words.get(comm), self.synonym_words.get(subsystem_id)]
        temp = self.synonym_replace(temp, tables2)
        # logger.debug("rebuild 结果: {} --> {}".format(question, temp))
        # 英文字符转小写
        temp = temp.lower()
        return temp

    def stop_words_delete2(self, fenci_list, tables):
        """
        :param fenci_list:
        :param tables:
        :return:
        """
        for table in tables:
            if not table:
                continue
            for word in table:
                fenci_list = [_ for _ in fenci_list if _ != word]
        return fenci_list

    def gen_sentences(self, fenci_list, subsystem_id, comm):
        """根据停用词和同义词尽可能多地生成同义句
        :param fenci_list: 句子分词后的 词语列表   ['我', '要', '取钱']
        :param subsystem_id: 子系统
        :param comm: 公共子系统
        :return: 句子列表 [['要 取钱'], ['我 要 取钱'], ['我 想要 取钱']]
        """

        tables2 = self.synonym_words
        tables3 = self.stop_words
        new_fenci_list = []
        for i, w in enumerate(
                fenci_list):  # fenci_list -> new_fenci_list 保证顺序同时去重
            if w not in fenci_list[:i]:
                new_fenci_list.append(w)
        qs = set()
        # 导航类提取关键词语作为同义句 -- 适当扩展容易混淆的地点
        sentence = ''.join(new_fenci_list)
        for place in ('迎宾点', '贵宾区', '贵宾点', '理财区'):
            if place in sentence:
                qs.add(place)

        # 导航类提取关键词语 end
        def _gen(ss):
            # ss: list  ['我', '要', '取钱']
            qs.add(' '.join(ss))
            for table in tables2:
                if not table:
                    continue
                for word_list in table:  # (a,b,c,d) 同义词列表, d最长
                    holder = "!"
                    word_list = flatten(word_list)
                    word_list.sort(key=len, reverse=True)
                    s = ss
                    for word in word_list:
                        s = [_ if _ != word else holder for _ in s]
                    for word in word_list[::-1]:
                        qs.add(' '.join(
                            [_ if _ != holder else word for _ in s]))

        _gen(new_fenci_list)
        _gen(self.stop_words_delete2(new_fenci_list, tables3))
        return list(qs)


if __name__ == "__main__":
    pass
    # w = Words(clients.mysql)
    # w2 = Words(clients.mysql)
    # print(id(w))
    # print(id(w2))
    # print(w.reload())