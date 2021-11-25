from config import get_cfg
from collections import deque
from operator import itemgetter
import os
import pickle
import sys
sys.setrecursionlimit(10000000)

_getitem0 = itemgetter(0)


def hamming_distance(x, y):
    """Calculate the hamming distance (number of bits different) between the
    two integers given.
    >>> [hamming_distance(x, 15) for x in [0, 8, 10, 12, 14, 15]]
    [4, 3, 2, 2, 1, 0]
    """
    return bin(x ^ y).count('1')


def Levenshteind(a, b):
    # Generate the length of both strings, which would be relevant in calculating the Levenshtein distance
    lena, lenb = len(a), len(b)

    # Create a table for storing the solutions as done in dynamic programming
    storage = [[0 for x in range(lenb + 1)] for x in range(lena + 1)]

    # Use dynamic programming to solve the problem in a bottom up fashion, by solving sub-problems
    for i in range(lena + 1):
        for j in range(lenb + 1):

            # fill up the base cases where either the first string or second string has length 0
            # if the first string has a length of 0, then insertions equal to the length
            # of the second string were done to get from the first string to the second.
            if i == 0:
                storage[i][j] = j

                # if the second string has a length of 0, then deletions equal in number to the
                # length of the first string were done on the first string to get to the second string
            elif j == 0:
                storage[i][j] = i

                # if both strings have the same last characters, then no change was done on the last character
                # of the first string to get to the last character of the second string. Therefore, the number
                # of changes would be equal to the changes done on the remaining characters of the first string
                # to get to the remaining characters of the second
            elif a[i - 1] == b[j - 1]:
                storage[i][j] = storage[i - 1][j - 1]

                # if both strings have different last characters, then a change was done to get from the first
                # string to the second string. This change could either be an insertion, deletion or mutation,
                # but never three of them at the same time. The change done will be 1 plus the previous record
                # state of insertion, deletion or mutation
            else:
                storage[i][j] = 1 + min(storage[i][j - 1], storage[i - 1][j],
                                        storage[i - 1][j - 1])
    return storage[lena][lenb]


class BKtree(object):
    """BK-tree data structure that allows fast querying of matches that are
    "close" given a function to calculate a distance metric (e.g., Hamming
    distance or Levenshtein distance).
    Each node in the tree (including the root node) is a two-tuple of
    (item, children_dict), where children_dict is a dict whose keys are
    non-negative distances of the child to the current item and whose values
    are nodes.
    """
    def __init__(self, cfg, distance_func=Levenshteind):
        """Initialize a BKTree instance with given distance function
        (which takes two items as parameters and returns a non-negative
        distance integer). "items" is an optional list of items to add
        on initialization.
        >>> tree = BKTree(hamming_distance)
        >>> list(tree)
        []
        >>> tree.distance_func is hamming_distance
        True
        >>> tree = BKTree(hamming_distance, [])
        >>> list(tree)
        []
        >>> tree = BKTree(hamming_distance, [0, 4, 5])
        >>> sorted(tree)
        [0, 4, 5]
        """
        self.cfg = cfg
        self.distance_func = distance_func
        self.tree = None

    def builder(self, data=None):
        if os.path.exists(self.cfg.CORRECTION.BKTREE_PATH):
            with open(self.cfg.CORRECTION.BKTREE_PATH, 'rb') as f:
                self.tree = pickle.load(f)
        else:
            _add = self.add
            for item in data:
                _add(item)
            with open(self.cfg.CORRECTION.BKTREE_PATH, 'wb') as f:
                pickle.dump(self.tree, f)

    def add(self, item):
        """Add given item to this tree.
        >>> tree = BKTree(hamming_distance)
        >>> list(tree)
        []
        >>> tree.add(4)
        >>> sorted(tree)
        [4]
        >>> tree.add(15)
        >>> sorted(tree)
        [4, 15]
        """
        node = self.tree
        if node is None:
            self.tree = (item, {})
            return

        # Slight speed optimization -- avoid lookups inside the loop
        _distance_func = self.distance_func

        while True:
            parent, children = node
            distance = _distance_func(item, parent)
            node = children.get(distance)
            if node is None:
                children[distance] = (item, {})
                break

    def find(self, item, n):
        """Find items in this tree whose distance is less than or equal to n
        from given item, and return list of (distance, item) tuples ordered by
        distance.
        >>> tree = BKTree(hamming_distance)
        >>> tree.find(13, 1)
        []
        >>> tree.add(0)
        >>> tree.find(1, 1)
        [(1, 0)]
        >>> for item in [0, 4, 5, 14, 15]:
        ...     tree.add(item)
        >>> sorted(tree)
        [0, 0, 4, 5, 14, 15]
        >>> sorted(tree.find(13, 1))
        [(1, 5), (1, 15)]
        >>> sorted(tree.find(13, 2))
        [(1, 5), (1, 15), (2, 4), (2, 14)]
        >>> sorted(tree.find(0, 1000)) == [(hamming_distance(x, 0), x) for x in tree]
        True
        """
        if self.tree is None:
            return []

        candidates = deque([self.tree])
        found = []

        # Slight speed optimization -- avoid lookups inside the loop
        _candidates_popleft = candidates.popleft
        _candidates_extend = candidates.extend
        _found_append = found.append
        _distance_func = self.distance_func

        while candidates:
            candidate, children = _candidates_popleft()
            distance = _distance_func(candidate, item)
            if distance <= n:
                _found_append((distance, candidate))

            if children:
                lower = distance - n
                upper = distance + n
                _candidates_extend(c for d, c in children.items() if lower <= d <= upper)

        found.sort(key=_getitem0)
        return found

    def __iter__(self):
        """Return iterator over all items in this tree; items are yielded in
        arbitrary order.
        >>> tree = BKTree(hamming_distance)
        >>> list(tree)
        []
        >>> tree = BKTree(hamming_distance, [1, 2, 3, 4, 5])
        >>> sorted(tree)
        [1, 2, 3, 4, 5]
        """
        if self.tree is None:
            return

        candidates = deque([self.tree])

        # Slight speed optimization -- avoid lookups inside the loop
        _candidates_popleft = candidates.popleft
        _candidates_extend = candidates.extend

        while candidates:
            candidate, children = _candidates_popleft()
            yield candidate
            _candidates_extend(children.values())

    def __repr__(self):
        """Return a string representation of this BK-tree with a little bit of info.
        >>> BKTree(hamming_distance)
        <BKTree using hamming_distance with no top-level nodes>
        >>> BKTree(hamming_distance, [0, 4, 8, 14, 15])
        <BKTree using hamming_distance with 3 top-level nodes>
        """
        return '<{} using {} with {} top-level nodes>'.format(
            self.__class__.__name__,
            self.distance_func.__name__,
            len(self.tree[1]) if self.tree is not None else 'no',
        )


# This part of the code is a test function , that just tests a sample case
if __name__ == '__main__':
    cfg = get_cfg()
    text = ["".join(x.strip().split()) for x in open(cfg.BASE.FINE_WORD_FILE).readlines()][10000]
    tree = BKtree(cfg)
    import time
    s = time.time()
    tree.builder(text)
    print('build time = {}'.format(time.time() - s))
    test = [
        "我要买猫沙盆",  # 错字
        "窝要去医院",  # 错字
        "看见猫猫精神没有",  # 乱序
        "狗狗发了",    # 少字
        "maomi",     # 拼音
        "猫咪啦肚子怎么办",   
        "我家狗拉啦稀了",   # 多字
        "宠物托运该怎么办", 
        "狗狗偶尔尿xie怎么办",
        "狗老抽出怎么办",
        "看见猫，猫吃了张纸，怎么办",
        '这周末我要去配副眼睛',
        '我家猫猫感帽了',
        '随然今天很热',
        '传然给我',
        '呕土不止',
        '我生病了,咳数了好几天',
        '我想买哥苹果手机'
    ]
    min_t = 100
    max_t = 0
    all_t = []
    for x in test:
        s = time.time()
        res = tree.find(x, 2)
        t = time.time() - s
        all_t.append(t)
        if t < min_t:
            min_t = t
        if t > max_t:
            max_t = t
    print('Python version took min time {}, max time {}, avg time {}'.format(min_t, max_t, sum(all_t) / len(all_t)))

    from qa.tools.bktree.BKTree import BKTree
    bktree = BKTree()

    s = time.time()
    for i in text:
        bktree.add(i)
    print('build time = {}'.format(time.time() - s))
    
    min_t = 100
    max_t = 0
    all_t = []
    for i in test:
        s = time.time()
        res = bktree.search(i, 2)
        t = time.time() - s
        all_t.append(t)
        if t < min_t:
            min_t = t
        if t > max_t:
            max_t = t
    print('Python version took min time {}, max time {}, avg time {}'.format(min_t, max_t, sum(all_t) / len(all_t)))

