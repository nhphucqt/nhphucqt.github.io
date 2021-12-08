#include <bits/stdc++.h>

using namespace std;

const int N = 1e5+7;
int numNode, numEdge, source, dest;
vector<pair<int,int>> adj[N];
long long dist[N];
int trace[N];
bool inq[N];

long long dijkstra(int source, int dest) {
    memset(inq, false, (numNode+1) * sizeof(bool));
    memset(dist, 0x3f, (numNode+1) * sizeof(long long));
    memset(trace, 0, (numNode+1) * sizeof(int));
    dist[source] = 0;
    trace[source] = -1;
    inq[source] = true;
    while (true) {
        int u = -1;
        for (int v = 1; v <= numNode; ++v) if (inq[v]) {
            if (u == -1 || dist[u] > dist[v]) u = v;
        }
        if (u == -1) break;
        if (u == dest) 
            return dist[u];
        inq[u] = false;
        for (pair<int,int> tmp : adj[u]) {
            int v = tmp.first;
            int w = tmp.second;
            if (dist[v] > dist[u] + w) {
                dist[v] = dist[u] + w;
                trace[v] = u;
                inq[v] = true;
            }
        }
    }
    return -1;
}

int main() {
    cin.tie(nullptr)->sync_with_stdio(false);
    cin >> numNode >> numEdge;
    for (int i = 0; i < numEdge; ++i) {
        int x, y, w;
        cin >> x >> y >> w;
        adj[x].emplace_back(y,w);
        adj[y].emplace_back(x,w);
    }
    cin >> source >> dest;
    long long minPath = dijkstra(source, dest);
    cout << minPath << '\n';
    if (minPath != -1) {
        vector<int> path;
        for (int u = numNode; u != -1; u = trace[u]) {
            path.push_back(u);
        }
        for (int i = (int)path.size()-1; i >= 0; --i) {
            cout << path[i] << ' ';
        }
    }
    return 0;
}