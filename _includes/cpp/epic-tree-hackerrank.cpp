#include <bits/stdc++.h>

using namespace std;

struct HLD {
    int position, highest, last;
    HLD() { position = highest = 0; }
    HLD(int p, int h) {
        position = p;
        highest = h;
    }
};

const int N = 1e5+7;
const int LOG = 18;
int numNode, numQue;
int anc[N][LOG], depth[N], sz[N];
vector<int> adj[N], hldList;
HLD hld[N];

struct segtree {
    int size;
    vector<long long> lazy, val;
    vector<pair<long long, long long>> node;
    void init(int n) {
        size = 1;
        while(size < n) size <<= 1;
        node.assign(size<<1, make_pair(0,0));
        val.assign(size<<1,0);
        lazy.assign(size<<1,0);
        build(1,0,size-1);
    }
    void build(int id, int l, int r) {
        if (l == r) {
            if (l < numNode) {
                val[id] = sz[hldList[l]];
            }
            return;
        }
        int m = l+r>>1;
        build(id<<1,l,m);
        build(id<<1|1,m+1,r);
        val[id] = val[id<<1] + val[id<<1|1];
    }
    void pushDown(int id, int len) {
        if (lazy[id] == 0) return;
        for (int i = id<<1; i <= (id<<1|1); ++i) {
            lazy[i] += lazy[id];
            node[i].first += val[i] * lazy[id];
            node[i].second += lazy[id] * (len>>1);
        }
        lazy[id] = 0;
    }
    void add(int id, int l, int r, int x, int y, int k) {
        if (l>y||r<x) return;
        if (l>=x&&r<=y) {
            node[id].first += val[id] * k;
            node[id].second += 1LL * k * (r-l+1);
            lazy[id] += k;
            return;
        }
        pushDown(id, r-l+1);
        int m = l+r>>1;
        add(id<<1,l,m,x,y,k);
        add(id<<1|1,m+1,r,x,y,k);
        node[id].first = node[id<<1].first + node[id<<1|1].first;
        node[id].second = node[id<<1].second + node[id<<1|1].second;
    }
    void add(int x, int y, int k) {
        if (x > y) return;
        add(1,0,size-1,x,y,k);
    }
    long long sum(int id, int l, int r, int x, int y) {
        if (l>y||r<x) return 0;
        if (l>=x&&r<=y) return node[id].first;
        pushDown(id, r-l+1);
        int m = l+r>>1;
        return sum(id<<1,l,m,x,y) + sum(id<<1|1,m+1,r,x,y);
    }
    long long sum(int x, int y) {
        return sum(1,0,size-1,x,y);
    }
    long long sum2(int id, int l, int r, int x, int y) {
        if (l>y||r<x) return 0;
        if (l>=x&&r<=y) return node[id].second;
        pushDown(id, r-l+1);
        int m = l+r>>1;
        return sum2(id<<1,l,m,x,y) + sum2(id<<1|1,m+1,r,x,y);
    }
    long long sum2(int x, int y) {
        return sum2(1,0,size-1,x,y);
    }
} st;

void dfs(int u, int p) {
    sz[u] = 1;
    for (int i = 1; i < LOG; ++i) {
        anc[u][i] = anc[anc[u][i-1]][i-1];
    }
    for (int v : adj[u]) if (v != p) {
        anc[v][0] = u;
        depth[v] = depth[u] + 1;
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
    if (bigNode != -1) {
        dfsHLD(bigNode, u, highest);
        for (int v : adj[u]) {
            if (v != p && v != bigNode) {
                dfsHLD(v, u, v);
            }
        }
    }
    hld[u].last = hldList.size();
}

void buildHLD() {
    depth[0] = -1;
    dfs(1,-1);
    dfsHLD(1,-1,1);
    // subNode of u in [hld[u].position, hld[u].last)
    st.init(numNode);
}

int lca(int u, int v) {
    if (depth[u] < depth[v]) swap(u,v);
    for (int i = LOG-1; i >= 0; --i) {
        if (depth[anc[u][i]] >= depth[v]) {
            u = anc[u][i];
        }
    }
    if (u == v) return u;
    for (int i = LOG-1; i >= 0; --i) {
        if (anc[u][i] != anc[v][i]) {
            u = anc[u][i];
            v = anc[v][i];
        }
    }
    return anc[u][0];
}

void add(int u, int v, int k) {
    int par = lca(u,v);
    for (; hld[u].highest != hld[par].highest; u = anc[hld[u].highest][0]) {
        st.add(hld[hld[u].highest].position, hld[u].position, k);
    }
    st.add(hld[par].position, hld[u].position, k);
    for (; hld[v].highest != hld[par].highest; v = anc[hld[v].highest][0]) {
        st.add(hld[hld[v].highest].position, hld[v].position, k);
    }
    st.add(hld[par].position+1, hld[v].position, k);
}

long long sum(int u, int num) {
    long long s = 0;
    for (; hld[u].highest != hld[1].highest; u = anc[hld[u].highest][0]) {
        s += st.sum2(hld[hld[u].highest].position, hld[u].position);
    }
    s += st.sum2(hld[1].position, hld[u].position);
    return s * num;
}

long long getVal(int u) {
    long long val = st.sum(hld[u].position, hld[u].last-1);
    if (u != 1) val += sum(anc[u][0], sz[u]);
    return val;
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
    cin >> numQue;
    while (numQue--) {
        int t; cin >> t;
        if (t == 1) {
            int u, v, k;
            cin >> u >> v >> k;
            add(u, v, k);
        }
        else {
            int u; cin >> u;
            cout << getVal(u) % 10009 << '\n';
        }
    }
    return 0;
}