#!/usr/bin/python
import time

from config import get_cfg
from qa.queryUnderstanding.querySegmentation.words import Words

from qa.tools import flatten


# This class represents a node in the directed acyclic word graph (DAWG). It
# has a list of edges to other nodes. It has functions for testing whether it
# is equivalent to another node. Nodes are equivalent if they have identical
# edges, and each identical edge leads to identical states. The __hash__ and
# __eq__ functions allow it to be used as a key in a python dictionary.
class DawgNode:
    NextId = 0

    def __init__(self):
        self.id = DawgNode.NextId
        DawgNode.NextId += 1
        self.final = False
        self.edges = {}

    def __str__(self):
        arr = []
        if self.final:
            arr.append("1")
        else:
            arr.append("0")

        for (label, node) in self.edges.items():
            arr.append(label)
            arr.append(str(node.id))

        return "_".join(arr)

    def __hash__(self):
        return self.__str__().__hash__()

    def __eq__(self, other):
        return self.__str__() == other.__str__()


class Dawg:
    def __init__(self):
        self.previousWord = ""
        self.root = DawgNode()

        # Here is a list of nodes that have not been checked for duplication.
        self.uncheckedNodes = []

        # Here is a list of unique nodes that have been checked for
        # duplication.
        self.minimizedNodes = {}

    def insert(self, word):
        if word < self.previousWord:
            raise Exception("Error: Words must be inserted in alphabetical order.")

        # find common prefix between word and previous word
        commonPrefix = 0
        for i in range(min(len(word), len(self.previousWord))):
            if word[i] != self.previousWord[i]: 
                break
            commonPrefix += 1

        # Check the uncheckedNodes for redundant nodes, proceeding from last
        # one down to the common prefix size. Then truncate the list at that
        # point.
        self._minimize(commonPrefix)

        # add the suffix, starting from the correct node mid-way through the
        # graph
        if len(self.uncheckedNodes) == 0:
            node = self.root
        else:
            node = self.uncheckedNodes[-1][2]

        for letter in word[commonPrefix:]:
            nextNode = DawgNode()
            node.edges[letter] = nextNode
            self.uncheckedNodes.append((node, letter, nextNode))
            node = nextNode

        node.final = True
        self.previousWord = word

    def finish(self):
        # minimize all uncheckedNodes
        self._minimize(0)

    def _minimize(self, downTo):
        # proceed from the leaf up to a certain point
        for i in range(len(self.uncheckedNodes) - 1, downTo - 1, -1):
            (parent, letter, child) = self.uncheckedNodes[i]
            if child in self.minimizedNodes:
                # replace the child with the previously encountered one
                parent.edges[letter] = self.minimizedNodes[child]
            else:
                # add the state to the minimized nodes.
                self.minimizedNodes[child] = child
            self.uncheckedNodes.pop()

    def lookup(self, word):
        node = self.root
        for letter in word:
            if letter not in node.edges: 
                return False
            node = node.edges[letter]

        return node.final

    def nodeCount(self):
        return len(self.minimizedNodes)

    def edgeCount(self):
        count = 0
        for node in self.minimizedNodes:
            count += len(node.edges)
        return count


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
    QUERY = ['fasiqi']
    dawg = Dawg()
    WordCount = 0
    words.sort()
    start = time.time()
    for word in words:
        WordCount += 1
        dawg.insert(word)
        if (WordCount % 100) == 0: 
            print("%dr" % WordCount)
    dawg.finish()
    print("Dawg creation took %g s" % (time.time() - start))

    EdgeCount = dawg.edgeCount()
    print("Read %d words into %d nodes and %d edges" %
          (WordCount, dawg.nodeCount(), EdgeCount))
    print("This could be stored in as little as %d bytes" % (EdgeCount * 4))

    for word in QUERY:
        start = time.time()
        res = dawg.lookup(word)
        print("Dawg lookup took %g s" % (time.time() - start))
        if not res:
            print("%s not in dictionary." % word)
        else:
            print("%s is in the dictionary." % word)