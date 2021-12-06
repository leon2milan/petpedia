#ifndef BKTree_h
#define BKTree_h

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

using std::string;
using std::vector;
using std::basic_string;

struct Node
{
    string word;
    size_t distance;
    Node* leftChild;
    Node* rightSibling;
};

class BKTree
{
private:
    Node* root;
    Node* createNode(string w, size_t d);
    int min(int a, int b, int c);
    size_t levenshteinDistance(string w1, string w2);
    void recursiveSearch(Node* node, vector<string>& suggestions, string w,
                         size_t t, bool& wordFound);
    bool inRange(size_t curDist, size_t minDist, size_t maxDist);
    // void printSuggestions(vector<string>& suggestions, bool wordFound);
public:
    BKTree();
    ~BKTree();
    bool add(string w);
    vector<string> search(string w, int t);
};

#endif /* BKTree_h */
