import os
import pickle

from config import get_cfg
from qa.queryUnderstanding.querySegmentation.words import Words
from qa.tools import flatten
import sys

sys.setrecursionlimit(10000)


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.word = str


class Trie(object):
    """实现Tried树的类
    Attributes:
        __root: Node类型，Tried树根节点
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = TrieNode()

    def add_word(self, word):
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_word = True
        node.word = word

    def kDistance(self, target, k):
        n = len(target)
        dp = [i for i in range(n + 1)]
        result = []

        self.find(self.root, target, k, dp, result)
        return result

    def find(self, node, target, k, dp, result):
        n = len(target)

        if node.is_word and dp[n] <= k:
            result.append(node.word)

        next = [0 for i in range(n + 1)]
        for c in node.children:
            next[0] = dp[0] + 1
            for i in range(1, n + 1):
                if target[i - 1] == c:
                    # print(target[i - 1], c)
                    next[i] = min(dp[i - 1], min(dp[i] + 1, next[i - 1] + 1))
                else:
                    next[i] = min(dp[i - 1] + 1, dp[i] + 1, next[i - 1] + 1)

            self.find(node.children[c], target, k, next, result)

    def save(self, name):
        with open(
                os.path.join(self.cfg.BASE.MODEL_PATH, "embedding",
                             name + '_trie.pkl'), 'wb') as f:
            pickle.dump(self.root, f)

    def load(self, name):
        with open(
                os.path.join(self.cfg.BASE.MODEL_PATH, "embedding",
                             name + '_trie.pkl'), 'rb') as f:
            self.root = pickle.load(f)


if __name__ == "__main__":

    cfg = get_cfg()
    from qa.queryUnderstanding.queryReformat.queryCorrection.pinyin import Pinyin

    from functools import reduce
    merge_all = reduce(lambda a, b: dict(a, **b),
                       Words(cfg).get_specializewords.values())
    words = flatten([
        "".join(Pinyin.get_pinyin_list(v))
        for value in merge_all.values() for v in value if v
    ])
    string = 'fasiqi'
    tree = Trie(cfg)
    for word in words:
        tree.add_word(word)

    import time
    s = time.time()
    res = tree.kDistance(string, 1)
    print(res, time.time() - s)
