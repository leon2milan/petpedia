import pandas as pd
import time
import math
import os
import re

from qa.queryUnderstanding.querySegmentation import Segmentation, Words
from config import get_cfg


def load_word_dict():
    word_dict = {}
    stop_words = {}

    if os.path.exists("../small_seg.dic"):
        with open("../small_seg.dic", "r", encoding="utf8") as fr:
            word_dict = {
                word.strip().split(" ")[0]: word.strip().split(" ")[-1]
                for word in fr
            }

    if os.path.exists("../stopwords.txt"):
        with open("../stopwords.txt", encoding="utf8") as fr:
            stop_words = set([word.strip() for word in fr])

    return word_dict, stop_words


class Config(object):
    """
    各阈值设定
    words_length: 新词字数默认个字
    pmi_limit: 凝固度阈值
    left_entropy_limit: 左熵阈值
    right_entropy_limit: 右熵阈值
    word_frequency: 词频阈值
    """
    words_length = 5
    pmi_limit = 1.5
    left_entropy_limit = 1
    right_entropy_limit = 1
    word_frequency_limit = 5
    word_dict, stop_words = load_word_dict()


class ContentHandler(object):
    def __init__(self, content):
        """
        length: 文本字数
        book_content: 正序文本
        r_book_content: 倒序文本
        tire_tree: 正序文本 tire树
        r_tire_tree: 倒序文本 tire树
        """
        self.length = 0
        self.book_content, self.r_book_content = self._preprocess(
            content=content)

        self.tire_tree = TireTree(self.book_content)
        self.r_tire_tree = TireTree(self.r_book_content)

    def pmi(self, char_node):
        """计算凝固度"""
        p_x_y = char_node.count / self.length
        px_py_list = []

        # 枚举所有组成词的情况, 并取最大概率值
        for i in range(1, len(char_node.name)):
            px = self.tire_tree.search_node(
                char_node.name[:i]).count / self.length
            py = self.tire_tree.search_node(
                char_node.name[i:]).count / self.length
            px_py_list.append(px * py)

        px_py = max(px_py_list)
        p = math.log10(p_x_y / px_py)
        return p

    def left_entropy(self, char_node):
        """计算左熵"""
        r_char_node = self.r_tire_tree.search_node(char_node.name[::-1])
        father_set = r_char_node.child
        le = 0
        for father_name, father_node in father_set.items():
            p_father = father_node.count / r_char_node.child_counts
            p = p_father * math.log10(p_father)
            le += p
        return -le

    def right_entropy(self, char_node):
        """计算右熵"""
        child_set = self.tire_tree.search_node(char_node.name).child
        re = 0
        for child_name, child_node in child_set.items():
            p_child = child_node.count / char_node.child_counts
            p = p_child * math.log10(p_child)
            re += p
        return -re

    def word_frequency(self, char_node):
        """计算词频"""
        return char_node.count

    def get_words(self, node, layer, res_data):
        if layer >= self.tire_tree.layer - 1:
            return

        for c_name, c_node in node.child.items():
            # 递归
            self.get_words(c_node, layer + 1, res_data)

            # 纯小写英文及纯数字过滤
            if (c_name.encode("utf8").isalpha()
                    and c_name.encode("utf8").islower()
                ) or c_name.encode("utf8").isdigit():
                continue
            # 词典过滤
            is_continue = self.word_dict_filter(c_name)
            if not is_continue:
                continue
            # 阈值过滤
            plrf = self.limit_filter(c_node)
            if not plrf:
                continue

            res_data.append(plrf)

    def limit_filter(self, node):
        wf = node.count
        if wf < Config.word_frequency_limit:
            return False

        pmi = self.pmi(node)
        if pmi < Config.pmi_limit:
            return False

        le = self.left_entropy(node)
        if le < Config.left_entropy_limit:
            return False

        re = self.right_entropy(node)
        if re < Config.right_entropy_limit:
            return False

        return [node.name, wf, pmi, le, re]

    def word_dict_filter(self, chars):
        """词典过滤"""
        for char in self.permutation(1, chars):
            # 过滤掉已经收录于LAC的词
            if Config.word_dict.get(char, None) is not None:
                return False
            # 过滤停用词
            if char in Config.stop_words:
                return False
        return True

    def permutation(self, start_size, char):
        """字符排列组合"""
        for size in range(start_size, len(char) + 1):
            for index in range(len(char) + 1 - size):
                yield char[index:index + size]

    def _preprocess(self, content):
        """返回正序文本及倒序文本列表, 按符号拆分"""
        content_list = re.split("[^\u4E00-\u9FFFa-zA-Z0-9]", content)
        r_content_list = re.split("[^\u4E00-\u9FFFa-zA-Z0-9]", content[::-1])

        if not self.length:
            self.length = sum([len(i) for i in content_list if i])

        return content_list, r_content_list


class Node(object):
    def __init__(self, name, father):
        """
        节点名称
        节点出现次数
        父节点
        子节点列表
        未去重子集量
        """
        self.name = name
        self.count = 0
        self.father = father
        self.child = {}
        self.child_counts = 0


class TireTree(object):
    def __init__(self, content, layer_num=Config.words_length + 1, step=1):
        """
        字典树对象
        layer_num: 字典树层数
        content: 构建字典树的字符串
        """
        self.content = content
        self.root = Node("ROOT", None)
        self.layer = layer_num
        self.step = step
        self.word_counts = 0

        self.build_tree()

    def build_tree(self):
        # 按层构建字典树, layer 从1开始, 表示第一层, 第0层为根节点
        for layer in range(1, self.layer + 1):
            # 创建切割窗口对象
            char_session = CharSession(size=layer, step=self.step)
            for char in char_session.split_char(self.content):
                # 判断 char 是否在该层 没有节点则添加, 有则更新
                if not self.search_node(char, layer):
                    self.add_node(char, layer)
                else:
                    self.update_node(char, layer)

    def add_node(self, char, layer=None):
        """
        在指定层添加指定字符串节点
        """
        if not layer:
            layer = len(char)

        # 创建节点对象
        if layer == 1:
            father = self.root
        else:
            father = self.search_node(char[:-1])

        node = Node(name=char, father=father)
        node.count = 1

        # 将此节点挂入tire树
        father.child[char] = node
        father.child_counts += 1
        return True

    def del_node(self, char, layer=None):
        pass

    def update_node(self, char, layer=None):
        """
        更新指定节点信息
        """
        if not layer:
            layer = len(char)

        node = self.search_node(char, layer)
        node.count += 1
        node.father.child_counts += 1
        return True

    def search_node(self, char, layer=None):
        """
        指定字符串, 指定层 查找节点是否存在
        """
        if char == "ROOT":
            return self.root
        elif not layer:
            layer = len(char)

        node = self.root
        for layer_index in range(1, layer + 1):
            node = node.child.get(char[:layer_index], None)
            if not node:
                return None

        return node


class CharSession(object):
    def __init__(self, size, step=1):
        """
        窗口对象
        size: 窗口大小
        step: 移动步长
        """
        self.size = size
        self.step = step

    def split_char(self, content):
        """
        按指定窗口大小及步长切割文本
        """
        for seq in content:
            while seq:
                if len(seq) >= self.size:
                    yield seq[:self.size]
                    seq = seq[self.step:]
                else:
                    break


def find_word(file_like):
    """
    file_like: 文本内容或文本路径
    """
    if isinstance(file_like, str) and os.path.exists(file_like):
        with open(file_like, encoding="utf8") as fr:
            content = fr.read().replace(' ', '')
    else:
        content = file_like

    content_handler = ContentHandler(content)
    words = []

    for child_node in content_handler.tire_tree.search_node(
            "ROOT").child.values():
        content_handler.get_words(child_node, layer=1, res_data=words)

    return words


if __name__ == '__main__':
    import os
    from qa.tools import flatten
    stime = time.time()
    cfg = get_cfg()
    seg = Segmentation(cfg)

    specialize = Words(cfg).get_specializewords
    stopwords = Words(cfg).get_stopwords
    qa = pd.read_table(cfg.BASE.CHAR_FILE, header=None, names=['content'])
    qa['content'] = qa['content'].apply(
        lambda x: "".join(x.split())).drop_duplicates()
    qa = qa['content'].drop_duplicates().values.tolist()

    # word = pd.DataFrame(find_word("\n".join(map(str, qa[:600000]))),
    #                     columns=[
    #                         'word', 'term freq', 'aggregation coefficien',
    #                         'left entropy', 'right entropy'])
    # word.to_csv(os.path.join(cfg.BASE.DATA_PATH, 'dictionary/segmentation/new_word_1.csv'), index=False)
    # etime = time.time()
    # print("ALL DONE! 耗时 {} s".format(etime - stime))

    word = pd.DataFrame(find_word("\n".join(map(str, qa[600000:]))),
                        columns=[
                            'word', 'term freq', 'aggregation coefficien',
                            'left entropy', 'right entropy'
                        ])
    word.to_csv(os.path.join(cfg.BASE.DATA_PATH,
                             'dictionary/segmentation/new_word_2.csv'),
                index=False)
    etime = time.time()
    print("ALL DONE! 耗时 {} s".format(etime - stime))

    import math
    import copy
    from qa.queryUnderstanding.preprocess.preprocess import trans2simple

    def get_score(row):
        el = row['left entropy']
        er = row['right entropy']
        pmi = row['aggregation coefficien']
        lh = er * math.log(el / math.fabs(el - er)) + el * math.log(
            er / math.fabs(el - er))
        return pmi + lh

    new1 = pd.read_csv(
        os.path.join(cfg.BASE.DATA_PATH,
                     'dictionary/segmentation/new_word_1.csv'))
    new2 = pd.read_csv(
        os.path.join(cfg.BASE.DATA_PATH,
                     'dictionary/segmentation/new_word_2.csv'))
    new = pd.concat([new1, new2])
    if os.path.exists(cfg.DICTIONARY.CUSTOM_WORDS):
        old = pd.read_csv(os.path.join(cfg.DICTIONARY.CUSTOM_WORDS))
        new = pd.concat([new, old]).drop_duplicates().reset_index(drop=True)
    word = copy.deepcopy(new)

    from functools import reduce
    entity_word = flatten([[k, v]
                           for k, v in reduce(lambda a, b: dict(a, **b),
                                              specialize.values()).items()])

    word['word'] = word['word'].progress_apply(
        lambda x: trans2simple(x)).dropna()
    word['word'] = word['word'].progress_apply(
        lambda x: re.sub('\（.*\）', '', x))
    word = word.drop_duplicates().dropna()
    word['cut'] = word['word'].progress_apply(
        lambda x: list(seg.cut(x, mode='pos')))
    word['len'] = word['cut'].progress_apply(lambda x: len(x[0]))
    word['stopwords'] = word['cut'].progress_apply(
        lambda x: "".join([y for y in x[0] if y in stopwords]))

    word = word[word['stopwords'] == '']
    word = word[~word['word'].
                isin([re.sub('\（.*\）', '', x) for x in entity_word])]
    word = word[(word['len'] > 1)]
    word = word[~word['word'].str.startswith('月')]
    word = word[~word['word'].str.startswith('都')]
    word = word[~word['word'].str.startswith('总')]
    word = word[~word['word'].str.startswith('必')]
    word = word[~word['word'].str.startswith('已')]
    word = word[~word['word'].str.startswith('岁')]
    word = word[~word['word'].str.startswith('请')]
    word = word[~word['word'].str.startswith('还是')]
    word = word[~word['word'].str.endswith('都')]
    word = word[~word['word'].str.endswith('了')]
    exclude = [
        '定期给狗狗', '定期给猫咪', '犬窝咳', '分为急性', '一般分为', '号楼', '度不够', '体成熟', '应及时',
        '受惊吓', '多大开始', '带著', '做好防', '到六个月', '到一定程度', '你可以把', '我相信很多', '部肿胀',
        '着身子', '没法正常', '期过后', '喵想', '座最适合养', '城小区', '大便软', '天中午', '将会增加',
        '层毛发', '得到处都是', '飞机耳', '骨骨折', '齿龈红', '凯恩更', '喵糖', '毯子或者', '钱买', '天之内',
        '分成两', '中加入', '中含有一种', '中含有', '中还含有', '中获得', '左右即可', '期间不建议', '中富含',
        '中可能含有', '中含有大量', '中含有一种', '中含有多种', '中都含有'
    ]
    word = word[~word['word'].isin(exclude)]

    word['word'].drop_duplicates().sort_values().to_csv(os.path.join(
        cfg.BASE.DATA_PATH, 'dictionary/segmentation/new_word.csv'),
                                                        index=False)
