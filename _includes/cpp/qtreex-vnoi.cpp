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
 
struct Edge {
    int x, y, w;
    Edge() { x = y = w = 0; }
    Edge(int _x, int _y, int _w) {
        x = _x; y = _y; w = _w;
    }
};
 
struct segtree {
    int size;
    vector<int> maxs, mins, lazy;
    void init(int numNode) {
        size = 1;
        while (size < numNode) size<<=1;
        maxs.assign(size<<1,0);
        mins.assign(size<<1,0);
        lazy.assign(size<<1,1);
    }
    void pushDown(int id) {
        if (lazy[id] != 1) {
            lazy[id<<1] *= -1;
            lazy[id<<1|1] *= -1;
            maxs[id<<1] *= -1;
            maxs[id<<1|1] *= -1;
            mins[id<<1] *= -1;
            mins[id<<1|1] *= -1;
            swap(maxs[id<<1],mins[id<<1]);
            swap(maxs[id<<1|1],mins[id<<1|1]);
            lazy[id] = 1;
        }
    }
    void update(int id, int l, int r, int i, int k) {
        if (l>i||r<i) return;
        if (l==r) {
            mins[id] = maxs[id] = k;
            return;
        }
        pushDown(id);
        int m = l+r>>1;
        update(id<<1,l,m,i,k);
        update(id<<1|1,m+1,r,i,k);
        mins[id] = min(mins[id<<1],mins[id<<1|1]);
        maxs[id] = max(maxs[id<<1],maxs[id<<1|1]);
    }
    void update(int i, int k) {
        update(1,0,size-1,i,k);
    }
    void Negate(int id, int l, int r, int x, int y) {
        if (l>y||r<x) return;
        if (l>=x&&r<=y) {
            lazy[id] *= -1;
            mins[id] *= -1;
            maxs[id] *= -1;
            swap(mins[id], maxs[id]);
            return;
        }
        pushDown(id);
        int m = l+r>>1;
        Negate(id<<1,l,m,x,y);
        Negate(id<<1|1,m+1,r,x,y);
        mins[id] = min(mins[id<<1],mins[id<<1|1]);
        maxs[id] = max(maxs[id<<1],maxs[id<<1|1]);
    }
    void Negate(int x, int y) {
        if (x > y) return;
        Negate(1,0,size-1,x,y);
    }
    int getMax(int id, int l, int r, int x, int y) {
        if (l>y||r<x) return INT_MIN;
        if (l>=x&&r<=y) return maxs[id];
        pushDown(id);
        int m = l+r>>1;
        return max(getMax(id<<1,l,m,x,y),getMax(id<<1|1,m+1,r,x,y));
    }
    int getMax(int x, int y) {
        if (x > y) return INT_MIN;
        return getMax(1,0,size-1,x,y);
    }
} st;
 
const int N = 1e4+7;
const int LOG = 15;
int numNode, sz[N], anc[N][LOG], depth[N];
vector<int> adj[N], hldList;
vector<Edge> edge;
HLD hld[N];
 
void dfs(int u, int p) {
    for (int i = 1; i < LOG; ++i) {
        anc[u][i] = anc[anc[u][i-1]][i-1];
    }
    sz[u] = 1;
    for (int i : adj[u]) {
        int v = u ^ edge[i].x ^ edge[i].y;
        if (v != p) {
            anc[v][0] = u;
            depth[v] = depth[u] + 1;
            dfs(v, u);
            sz[u] += sz[v];
        }
    }
}
 
void dfsHLD(int u, int p, int highest) {
    hld[u] = HLD(hldList.size(), highest);
    hldList.push_back(u);
    int bigNode = -1;
    for (int i : adj[u]) {
        int v = u ^ edge[i].x ^ edge[i].y;
        if (v != p) {
            if (bigNode == -1 || sz[bigNode] < sz[v]) {
                bigNode = v;
            }
        }
    }
    if (bigNode == -1) return;
    dfsHLD(bigNode, u, hld[u].highest);
    for (int i : adj[u]) {
        int v = u ^ edge[i].x ^ edge[i].y;
        if (v!=p && v!=bigNode) {
            dfsHLD(v, u, v);
        }
    }
}
 
void buildHLD() {
    depth[0] = -1;
    dfs(1,0);
    dfsHLD(1, 0, 1);
    st.init(numNode);
    for (int i = 0; i < edge.size(); ++i) {
        if (sz[edge[i].x] < sz[edge[i].y]) swap(edge[i].x,edge[i].y);
        st.update(hld[edge[i].y].position, edge[i].w);
    }
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
 
void maximize(int&x, int y) {
    if (x < y) x = y;
}
 
void update(int id, int val) {
    st.update(hld[edge[id].y].position, val);
}

void Negate(int u, int v) {
    int k = lca(u,v);
    for (; hld[u].highest != hld[k].highest; u = anc[hld[u].highest][0])
        st.Negate(hld[hld[u].highest].position, hld[u].position);
    st.Negate(hld[k].position+1, hld[u].position);
    for (; hld[v].highest != hld[k].highest; v = anc[hld[v].highest][0])
        st.Negate(hld[hld[v].highest].position, hld[v].position);
    st.Negate(hld[k].position+1, hld[v].position);
}

int getMax(int u, int v) {
    if (u == v) return 0;
    int k = lca(u,v), maxs = INT_MIN;
    for (; hld[u].highest != hld[k].highest; u = anc[hld[u].highest][0])
        maximize(maxs, st.getMax(hld[hld[u].highest].position, hld[u].position));
    maximize(maxs, st.getMax(hld[k].position+1, hld[u].position));
    for (; hld[v].highest != hld[k].highest; v = anc[hld[v].highest][0])
        maximize(maxs, st.getMax(hld[hld[v].highest].position, hld[v].position));
    maximize(maxs, st.getMax(hld[k].position+1, hld[v].position));
    return maxs;
}
 
int main() {
    cin.tie(nullptr)->sync_with_stdio(false);
    int nTest; cin >> nTest;
    while (nTest--) {
        cin >> numNode;
        for (int i = 1; i < numNode; ++i) {
            int x, y, w;
            cin >> x >> y >> w;
            adj[x].push_back(edge.size());
            adj[y].push_back(edge.size());
            edge.emplace_back(x, y, w);
        }
        buildHLD();
        string t;
        while ((cin >> t) && t != "DONE") {
            if (t == "CHANGE") {
                int id, val;
                cin >> id >> val;
                update(id-1, val);
            }
            else if (t == "NEGATE") {
                int u, v;
                cin >> u >> v;
                Negate(u,v);
            }
            else {
                int u, v;
                cin >> u >> v;
                cout << getMax(u,v) << '\n';
            }
        }
        // reset
        hldList.clear();
        edge.clear();
        for (int i = 1; i <= numNode; ++i) {
            adj[i] = vector<int>();
        }
    }
    return 0;   
}