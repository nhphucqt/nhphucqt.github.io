#include <bits/stdc++.h>

using namespace std;

const int NODE = 100000;
const int EDGE = 100000;
int numNode, numEdge;
vector<int> adj[NODE];
int Time, num[NODE], low[NODE];
int numComp, compID[NODE];
stack<int> comp;

void dfs(int u) {
   num[u] = low[u] = ++Time;
   comp.push(u);
   for (int v : adj[u]) if (!compID[v]) {
       if (!num[v]) {
           dfs(v);
           low[u] = min(low[u], low[v]);
       }
       else {
           low[u] = min(low[u], num[v]);
       }
   }
   if (num[u] == low[u]) {
       numComp++;
       while (true) {
           int v = comp.top(); comp.pop();
           compID[v] = numComp;
           if (v == u) break;
       }
   }
}

void tarjan() {
   memset(num, 0, sizeof num);
   memset(low, 0, sizeof low);
   memset(compID, 0, sizeof compID);
   Time = numComp = 0;
   for (int i = 1; i <= numNode; ++i) {
       if (!num[i]) dfs(i);
   }
}

int main() {
   // các đỉnh có chỉ số từ 1 -> numNode
   cin >> numNode >> numEdge;
   for (int i = 0; i < numEdge; ++i) {
       int x, y; cin >> x >> y;
       adj[x].push_back(y);
   }  
   tarjan();
   cout << numComp << '\n';
   for (int i = 1; i <= numNode; ++i) {
       cout << i << " -> " << compID[i] << '\n';
   }
   return 0;
}
