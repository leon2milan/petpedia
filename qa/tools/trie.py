import re
import os
import pickle
from config import get_cfg
from qa.queryUnderstanding.querySegmentation.words import Words

from qa.tools import flatten
import sys
sys.setrecursionlimit(10000)


class Trie:
    def __init__(self, cfg, path=''):
        self.cfg = cfg
        self.strings = []
        self.dict = {}
        self.count_strings = 0
        self.path = path

    def add_word(self, string):
        trie = self

        for letter in string:
            trie.count_strings += 1
            if letter not in trie.dict:
                trie.dict[letter] = Trie(trie.path + letter)
            trie = trie.dict[letter]
        trie.count_strings += 1
        trie.strings.append(string)

    def __hash__(self):
        return id(self)

    def __repr__(self):
        answer = self.path + ":\n  count_strings:" + str(
            self.count_strings) + "\n  strings: " + str(
                self.strings) + "\n  dict:"

        def indent(string):
            p = re.compile("^(?!:$)", re.M)
            return p.sub("    ", string)

        for letter in sorted(self.dict.keys()):
            subtrie = self.dict[letter]
            answer = answer + indent("\n" + subtrie.__repr__())
        return answer

    def kDistance_old(self, target, k):
        # this method is 100 times slower than new way.
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

    def kDistance(self, string, max_edits):
        # This will be all trie/string pos pairs that we have seen
        found = set()
        # This will be all trie/string pos pairs that we start the next edit with
        start_at_edit = set()

        # At distance 0 we start with the base of the trie can match the start of the string.
        start_at_edit.add((self, 0))
        answers = []
        for edits in range(max_edits + 1):  # 0..max_edits inclusive
            start_at_next_edit = set()
            todo = list(start_at_edit)
            for trie, pos in todo:
                if (trie, pos) not in found:  # Have we processed this?
                    found.add((trie, pos))
                    if pos == len(string):
                        answers.extend(trie.strings)  # ANSWERS FOUND HERE!!!
                        # We have to delete from the other string
                        for next_trie in trie.dict.values():
                            start_at_next_edit.add((next_trie, pos))
                    else:
                        # This string could have an insertion
                        start_at_next_edit.add((trie, pos + 1))
                        for letter, next_trie in trie.dict.items():
                            # We could have had a a deletion in this string
                            start_at_next_edit.add((next_trie, pos))
                            if letter == string[pos]:
                                todo.append(
                                    (next_trie, pos + 1))  # we matched farther
                            else:
                                # Could have been a substitution
                                start_at_next_edit.add((next_trie, pos + 1))
            start_at_edit = start_at_next_edit
        return answers

    def save(self, name):
        with open(
                os.path.join(self.cfg.BASE.DATA_STRUCTURE.TRIE.SAVE_PATH,
                             name + '_trie.pkl'), 'wb') as f:
            pickle.dump(self.dict, f)

    def load(self, name):
        with open(
                os.path.join(self.cfg.BASE.DATA_STRUCTURE.TRIE.SAVE_PATH,
                             name + '_trie.pkl'), 'rb') as f:
            self.dict = pickle.load(f)


if __name__ == "__main__":

    cfg = get_cfg()
    from qa.queryUnderstanding.queryReformat.queryCorrection.pinyin import Pinyin

    from functools import reduce
    merge_all = reduce(lambda a, b: dict(a, **b),
                       Words(cfg).get_specializewords.values())
    words = flatten([
        "".join(Pinyin.get_pinyin_list(v)) for value in merge_all.values()
        for v in value if v
    ])
    string = 'hasiqi'
    tree = Trie(cfg)
    for word in words:
        tree.add_word(word)

    import time
    s = time.time()
    res = tree.kDistance('hasiqi', 1)
    print(res, time.time() - s)
