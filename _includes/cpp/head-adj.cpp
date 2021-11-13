#include <bits/stdc++.h>

using namespace std;

const int NODE = 100000;
const int EDGE = 100000;
int numNode, numEdge;
pair<int,int> edge[EDGE];
int head[NODE], adj[EDGE*2];

int main() {
   // các đỉnh có chỉ số từ 1 -> numNode
   cin >> numNode >> numEdge;
   for (int i = 0; i < numEdge; ++i) {
       cin >> edge[i].first >> edge[i].second;
       head[edge[i].first]++;
       head[edge[i].second]++;
   }
   for (int i = 1; i <= numNode+1; ++i) {
       head[i] += head[i-1];
   }
   for (int i = 0; i < numEdge; ++i) {
       adj[--head[edge[i].first]] = edge[i].second;
       adj[--head[edge[i].second]] = edge[i].first;
   }
   for (int u = 1; u <= numNode; ++u) {
       cout << u << ": ";
       for (int i = head[u]; i < head[u+1]; ++i) {
           cout << adj[i] << ' ';
       }
       cout << '\n';
   }
   return 0;
}
