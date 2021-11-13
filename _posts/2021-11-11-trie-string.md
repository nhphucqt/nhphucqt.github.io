---
title: Trie string
category: xu-li-xau
---

```cpp
const int NUM_CHAR = 26;
const char MIN_CHAR = 'a';

struct Trie {
    struct Node {
        bool isLeaf;
        int next[NUM_CHAR];
        Node() {
            isLeaf = false;
            memset(next, -1, sizeof next);
        }
    };
    int root;
    vector<Node> nodes;
    int newNode() {
        nodes.push_back(Node());
        return (int)nodes.size() - 1;
    }
    Trie() {
        nodes = vector<Node>();
        root = newNode();
    }
    void insert(const string &s) {
        int cur = root;
        for (int i = 0; i < s.size(); ++i) {
            int c = s[i] - MIN_CHAR;
            if (nodes[cur].next[c] == -1) {
                int newId = newNode();
                nodes[cur].next[c] = newId;
            }
            cur = nodes[cur].next[c];
        }
        nodes[cur].isLeaf = true;
    }
    int size() const {
        return nodes.size();
    }
    bool empty() const {
        return nodes.empty();
    }
    bool find(const string &s) const {
        int cur = root;
        for (int i = 0; i < s.size(); ++i) {
            int c = s[i] - MIN_CHAR;
            if (nodes[cur].next[c] == -1) 
                return false;
            cur = nodes[cur].next[c];
        }
        return nodes[cur].isLeaf;
    }
};
```