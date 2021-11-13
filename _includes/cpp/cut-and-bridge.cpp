#include <bits/stdc++.h>

using namespace std;

struct Edge {
   int x, y;
   bool isUsed, isBridge;
};

const int NODE = 100000;
const int EDGE = 100000;
int numNode, numEdge;
vector<int> adj[NODE];
vector<Edge> edge;
int Time, num[NODE], low[NODE], numChild[NODE];
bool isCut[NODE];

void dfs(int u) {
   num[u] = ++Time;
   low[u] = numNode + 1;
   for (int i : adj[u]) if (!edge[i].isUsed) {
       edge[i].isUsed = true;
       int v = edge[i].x ^ edge[i].y ^ u;
       if (!num[v]) {
           numChild[u]++;
           dfs(v);
           low[u] = min(low[u], low[v]);
           edge[i].isBridge |= num[u] < low[v];
           isCut[u] |= num[u] <= low[v];
       }
       else {
           low[u] = min(low[u], num[v]);
       }
   }
}

void findBridgeAndCut() {
   memset(num, 0, sizeof num);
   memset(low, 0, sizeof low);
   memset(numChild, 0, sizeof numChild);
   memset(isCut, false, sizeof isCut);
   Time = 0;
   for (int i = 1; i <= numNode; ++i) {
       if (!num[i]) {
           dfs(i);
           if (numChild[i] == 1) {
               isCut[i] = false;
           }
       }
   }
}

int main() {
   // các đỉnh có chỉ số từ 1 -> numNode
   cin >> numNode >> numEdge;
   for (int i = 0; i < numEdge; ++i) {
       int x, y; cin >> x >> y;
       adj[x].push_back(edge.size());
       adj[y].push_back(edge.size());
       edge.push_back({x,y});
   }
   findBridgeAndCut();
   cout << "Cut: ";
   for (int i = 1; i <= numNode; ++i) {
       if (isCut[i]) cout << i << ' ';
   }
   cout << '\n';
   cout << "Bridge:\n";
   for (int i = 0; i < numEdge; ++i) {
       if (edge[i].isBridge) {
           cout << edge[i].x << ' ' << edge[i].y << '\n';
       }
   }
   return 0;
}
