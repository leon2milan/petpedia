from qa.tools import setup_logger
from qa.queryUnderstanding.queryReformat.queryCorrection.pinyin import Pinyin
logger = setup_logger()


class Node:
    __slots__ = ["children", "word", "pinyin",
                 "isEnd"]  # todo 使用__slots__，它仅允许动态绑定()里面有的属性,该方法能节约内存消耗

    def __init__(self):
        """
        children：Node-obj, 子节点
        word: list, 子节点上的词列表
        pinyin: list, 子节点上的词拼音,索引和word同步
        isEnd: True 表示 word和pinyin不为空
        """
        self.children = None
        self.word = None
        self.pinyin = None
        self.isEnd = None


class DFA:
    roots = None

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    @classmethod
    def init(cls, mysql=None):
        sql = "select proper_noun_word, subsystem_id from way_proper_noun_word where is_deleted = 0 "
        error_code, error_msg, records = mysql.fetchall(
            sql
        )  # todo pymysql中返回查询集 (error_no:int,error_msg:str,result:tuple)
        if error_code != 0:  # 0 是请求成功状态码
            logger.error("mysql错误:{}".format(error_msg))
            return
        cls.roots = {}
        for record in records:  # 子系统id元组类型
            subsystem_id = record["subsystem_id"].lower()  # 子系统id
            if subsystem_id not in cls.roots:
                cls.roots[subsystem_id] = Node(
                )  # todo 将实例对象构建构建成一个字典 key: subsystem_id | value: Node()
            cls.add_word(cls.roots[subsystem_id],
                         record["proper_noun_word"])  # proper_noun_word中文字符

    @staticmethod
    def add_word(root, word):
        """添加word和pinyin"""
        node = root  # node就是一个Node()实例对象
        if len(word) < 2:
            return
        # flag = len(word) >= 3  # 大于3个字的词用声母来表示,比如 惠享存 --> h x c
        # flag = True
        pinyin_of_word = ""
        pinyin_list = Pinyin.get_pinyin_list(word)  # 处理后的拼音列表
        if len(pinyin_list) != len(word):  # 若处理后的拼音列表长度不等于传入的中文字符长度，则抛异常
            logger.warning("该方法不适用中英夹杂的词汇: {}".format(word))
            return

        for i in range(len(word)):
            # temp = DFA.get_pinyin(word[i])[0] if flag else DFA.get_pinyin(word[i])
            temp = pinyin_list[i]
            pinyin_of_word += temp
            temp = temp[0]
            if node.children is None:  # 子节点为空
                node.children = {temp: Node()}  # 则key：z  value: Node()实例对象
            elif temp not in node.children:  # 若 z 不在
                node.children[temp] = Node()
            node = node.children[temp]  # Node()
        if node.word:
            node.word.append(word)  # list, 子节点上的词列表
            node.pinyin.append(pinyin_of_word)  # list, 子节点上的词拼音,索引和word同步
        else:
            node.word = [word]  # node.word为none时，自己构建一个列表
            node.pinyin = [pinyin_of_word]
        node.isEnd = True  # True 表示 word和pinyin不为空

    @classmethod
    def replace(cls, msg, subsystem_id=None, flag=False):
        """ 拼音方式识别msg中的热词并返回替换后的msg, 拼音纠错
        :param flag: =True 需要拼音完全相同才匹配  =False 只要求长度<= 2 的句子进行拼音完全匹配，其余只做声母匹配
        :param subsystem_id: str
        :param msg:
        :return: msg, [hot_words]
        """
        temps = list(cls.find(msg, subsystem_id=subsystem_id, flag=flag))

        if len(temps) == 1:
            word, begin, end = temps[0]
            if word:
                msg = msg[:begin - 1] + word + msg[end:]
            return msg, [word]
        if len(temps) >= 2:
            (word1, b1, e1), (word2, b2, e2) = temps[:2]
            msg = msg[:b1 - 1] + word1 + msg[e1:b2 - 1] + word2 + msg[e2:]
            return msg, [word1, word2]
        return msg, []

    @classmethod
    def find(cls, msg, subsystem_id=None, flag=False):
        """ 从message(句子)中查找热词 优先查找长度较长的词语
        :param flag:
        :param subsystem_id: str
        :param msg: str 完整的句子
        :return: tuple (热词, 热词起始位置, 热词结束位置)
        """
        root = cls.roots.get(subsystem_id.lower())  # 获取Node() 那四个参数
        if not root:
            return
        msg_pinyin_list = Pinyin.get_pinyin_list(msg)  # 鼻音替换
        logger.info(f"pinyin匹配: { msg } -> { msg_pinyin_list }")
        num = len(msg_pinyin_list)  # 替换后pinyin list的长度
        if len(msg) != num:  # 不是纯中文的句子可能不是一个字符对应一个拼音  3+3 这类计算公式
            return
        i = 0
        while i < num:  # 遍历 解析对象msg
            p = root  # 获取Node() 那四个参数
            j = i
            i = i + 1
            raw_word_pinyin = ""
            raw_word = ""
            while j < num and p.children:  # index在num list中 | 存在子节点
                one_of_msg = msg[j]  # 句子中的一个字
                temp = msg_pinyin_list[j]  # 获取index下鼻音替换后的一个字
                # temp = dfa.get_pinyin(one_of_msg)
                # if temp in p.children or temp[0] in p.children[0]:
                if temp[0] in p.children:  # temp[0] 一般为声母   p.children 为字典，其键值为词声母
                    p = p.children[temp[0]]
                    j = j + 1
                else:
                    break
                raw_word_pinyin += temp
                raw_word += one_of_msg
            # i==j 表示只有一个字, i < j 表示两个字以上
            if i < j and p.word:
                # 上一部 --> 上一步
                if len(raw_word_pinyin) < 3 or flag:  # 对于词长小于3的词语需要拼音完全相等才算满足
                    temp_word = None
                    for index in range(len(p.word)):
                        print(raw_word_pinyin, "   ", p.pinyin[index])
                        if raw_word_pinyin == p.pinyin[index]:
                            temp_word = p.word[index]
                            if temp_word == raw_word:  # 拼音全等且文字全等, 表示完全命中,立即返回
                                temp_word = p.word[index]
                                break
                            else:  # 拼音全等但文字不等 只返回其中（最后）一个(每个满足该条件都返回的话不好处理)
                                temp_word = p.word[index]
                    if temp_word:
                        yield temp_word, i, j
                else:  # 对于词长 >=3 的词语只进行声母比较
                    temp_index = 0
                    distance = 200  # 表示距离无穷大
                    lraw = len(raw_word_pinyin)
                    for index in range(len(p.word)):
                        distance2 = abs(len(p.pinyin[index]) - lraw)
                        # print(distance2, p.word[index], p.pinyin[index])
                        if distance >= distance2:  # 找出拼音字符串长度距离最近的那项并最后返回
                            distance = distance2
                            temp_index = index
                            if p.pinyin[index] == raw_word_pinyin:
                                break
                    if distance != 200:
                        yield p.word[temp_index], i, j

    @classmethod
    def print(cls):
        """用于检测是否成功载入"""
        for k, root in cls.roots.items():
            logger.debug("------------子系统:{}--------------".format(k))
            cls._print(root)

    @staticmethod
    def _print(node, w=""):
        if node.isEnd:
            logger.debug("{:<10} {}  {}".format(w, node.pinyin, node.word))
        if node.children:
            for k, v in node.children.items():
                node = v
                DFA._print(node, w + k)


if __name__ == "__main__":

    sub_id = 'a'
    DFA.roots = {}
    root = DFA.roots[sub_id] = Node()
    DFA.add_word(root, '低柜')
    DFA.print()

    msg = '狗狗偶尔尿xie怎么办'
    msg_pinyin = Pinyin.get_pinyin_list(msg)

    print(DFA.replace(msg, subsystem_id=sub_id))
    print(list(DFA.find(msg, subsystem_id=sub_id)))