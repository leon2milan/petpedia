#include "library.h"

using std::basic_string;
using std::cout;
using std::endl;
using std::string;
using std::vector;

BKTree::BKTree() { root = NULL; }
BKTree::~BKTree() { delete root; }

Node *BKTree::createNode(string w, size_t d)
{
    Node *node = new Node();

    node->word = w;
    node->distance = d;
    node->leftChild = NULL;
    node->rightSibling = NULL;

    return node;
}

bool BKTree::add(string w)
{
    if (root == NULL)
    {
        root = createNode(w, -1);
        return false;
    }

    Node *curNode = root;
    Node *child;
    Node *newChild;
    size_t dist;

    while (1)
    {
        dist = levenshteinDistance(curNode->word, w);
        if (!dist)
            return false;
        child = curNode->leftChild;
        while (child)
        {
            if (child->distance == dist)
            {
                curNode = child;
                break;
            }
            child = child->rightSibling;
        }
        if (!child)
        {
            newChild = createNode(w, dist);
            newChild->rightSibling = curNode->leftChild;
            curNode->leftChild = newChild;
            break;
        }
    }
    return true;
}

vector<string> BKTree::search(string w, int t)
{
    vector<string> suggestions;
    bool wordFound = false;

    recursiveSearch(root, suggestions, w, t, wordFound);

    return suggestions;
}

void BKTree::recursiveSearch(Node *curNode, vector<string> &suggestions, string w,
                             size_t t, bool &wordFound)
{
    size_t curDist = levenshteinDistance(curNode->word, w);
    size_t minDist = curDist - t;
    size_t maxDist = curDist + t;

    if (!curDist)
    {
        wordFound = true;
        return;
    }
    if (curDist <= t)
        suggestions.push_back(curNode->word);

    Node *child = curNode->leftChild;
    if (!child)
        return;

    while (child)
    {
        if (inRange(child->distance, minDist, maxDist))
            recursiveSearch(child, suggestions, w, t, wordFound);

        child = child->rightSibling;
    }
}

bool BKTree::inRange(size_t curDist, size_t minDist, size_t maxDist)
{
    return (minDist <= curDist && curDist <= maxDist);
}


//https://en.wikipedia.org/wiki/Levenshtein_distance
size_t BKTree::levenshteinDistance(string w1, string w2)
{
    if (w1.length() == 0)
        return w2.length();
    if (w2.length() == 0)
        return w1.length();

    size_t n_w1 = w1.length();
    size_t n_w2 = w2.length();
    int cost;

    int d[n_w1 + 1][n_w2 + 1];

    for (int i = 0; i <= n_w1; i++)
        d[i][0] = i;
    for (int i = 0; i <= n_w2; i++)
        d[0][i] = i;

    for (int i = 1; i <= n_w1; i++)
    {
        for (int j = 1; j <= n_w2; j++)
        {

            cost = (w1[i - 1] == w2[j - 1]) ? 0 : 1;

            d[i][j] = min(d[i - 1][j] + 1,
                          d[i][j - 1] + 1,
                          d[i - 1][j - 1] + cost);
        }
    }

    return d[n_w1][n_w2];
}

int BKTree::min(int a, int b, int c)
{
    int min = a;
    if (b < min)
        min = b;
    if (c < min)
        min = c;

    return min;
}