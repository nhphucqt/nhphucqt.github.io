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

struct segtree {
    int size;
    vector<int> node;
    void init(int numNode) {
        size = 1;
        while (size < numNode) size<<=1;
        node.assign(size<<1,INT_MAX);
    }
    void update(int id, int l, int r, int i) {
        if (l>i||r<i) return;
        if (l==r) {
            node[id] = node[id] == INT_MAX ? i : INT_MAX;
            return;
        }
        int m = l+r>>1;
        update(id<<1,l,m,i);
        update(id<<1|1,m+1,r,i);
        node[id] = min(node[id<<1],node[id<<1|1]);
    }
    void update(int i) {
        update(1,0,size-1,i);
    }
    int getMin(int id, int l, int r, int x, int y) {
        if (l>y||r<x) return INT_MAX;
        if (l>=x&&r<=y) return node[id];
        int m = l+r>>1;
        return min(getMin(id<<1,l,m,x,y),getMin(id<<1|1,m+1,r,x,y));
    }
    int getMin(int x, int y) {
        if (x > y) return INT_MAX;
        return getMin(1,0,size-1,x,y);
    }
} st;

const int N = 1e5+7;
int numNode, numQue, sz[N], par[N];
vector<int> adj[N], hldList;
HLD hld[N];

void dfs(int u, int p) {
    sz[u] = 1;
    for (int v : adj[u]) if (v != p) {
        par[v] = u;
        dfs(v, u);
        sz[u] += sz[v];
    }
}

void dfsHLD(int u, int p, int highest) {
    hld[u] = HLD(hldList.size(), highest);
    hldList.push_back(u);
    int bigNode = -1;
    for (int v : adj[u]) if (v != p) {
        if (bigNode == -1 || sz[bigNode] < sz[v]) {
            bigNode = v;
        }
    }
    if (bigNode == -1) return;
    dfsHLD(bigNode, u, highest);
    for (int v : adj[u]) if (v!=p&&v!=bigNode) {
        dfsHLD(v, u, v);
    }
}

void buildHLD() {
    dfs(1,0);
    dfsHLD(1, 0, 1);
    st.init(numNode);
}

void minimize(int&x, int y) {
    if (x > y) x = y;
}

void update(int u) {
    st.update(hld[u].position);
}

int FindFirst(int u) {
    int mins = INT_MAX;
    for (; hld[u].highest != hld[1].highest; u = par[hld[u].highest])
        minimize(mins, st.getMin(hld[hld[u].highest].position, hld[u].position));
    minimize(mins, st.getMin(hld[1].position, hld[u].position));
    return mins == INT_MAX ? -1 : hldList[mins];
}

int main() {
    cin.tie(nullptr)->sync_with_stdio(false);
    cin >> numNode >> numQue;
    for (int i = 1; i < numNode; ++i) {
        int x, y;
        cin >> x >> y;
        adj[x].push_back(y);
        adj[y].push_back(x);
    }
    buildHLD();
    while (numQue--) {
        int t, u;
        cin >> t >> u;
        if (t==0) update(u);
        else cout << FindFirst(u) << '\n';
    }
    return 0;   
}