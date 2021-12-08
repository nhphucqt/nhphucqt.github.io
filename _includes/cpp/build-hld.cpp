#include <bits/stdc++.h>

using namespace std;

struct HLD {
    int position, highest;
    HLD() { position = highest = 0; }
    HLD(int p, int h) {
        position = p;
        highest = h;
    }
};

const int N = 1e5+7;
int numNode;
vector<int> adj[N], hldList;
int sz[N];
HLD hld[N];

void dfs(int u, int p) {
    sz[u] = 1;
    for (int v : adj[u]) if (v!=p) {
        dfs(v, u);
        sz[u] += sz[v];
    }
}

void dfsHLD(int u, int p, int highest) {
    hld[u] = HLD(hldList.size(), highest);
    hldList.push_back(u);
    int bigNode = -1;
    for (int v : adj[u]) if (v!=p) {
        if (bigNode == -1 || sz[bigNode] < sz[v]) {
            bigNode = v;
        }
    }
    if (bigNode == -1) return;
    dfsHLD(bigNode, u, highest);
    for (int v : adj[u]) if (v!=bigNode&&v!=p) {
        dfsHLD(v, u, v);
    }
}

void buildHLD() {
    dfs(1, 0);
    dfsHLD(1, 0, 1);
}

int main() {
    cin.tie(nullptr)->sync_with_stdio(false);
    cin >> numNode;
    for (int i = 1; i < numNode; ++i) {
        int x, y;
        cin >> x >> y;
        adj[x].push_back(y);
        adj[y].push_back(x);
    }
    buildHLD();
    return 0;
}