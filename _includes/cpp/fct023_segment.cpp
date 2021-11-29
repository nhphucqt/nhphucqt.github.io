#include <bits/stdc++.h>

using namespace std;

const int MAX = 16e5+7;
const int LOG = 21;
const int N = 2e5+7;
int numSeg, numQue, numNode;
pair<int,int> segs[N], ques[N];
vector<int> adj[MAX];
int anc[MAX][LOG];

void dfs(int u) {
    for (int i = 1; i < LOG; ++i) {
        anc[u][i] = anc[anc[u][i-1]][i-1];
    }
    for (int v : adj[u]) {
        anc[v][0] = u;
        dfs(v);
    }
}

int compress() {
    int tmpSize = 0;
    int* tmp[numSeg*2 + numQue*2];
    for (int i = 0; i < numSeg; ++i) {
        tmp[tmpSize++] = &segs[i].first;
        tmp[tmpSize++] = &segs[i].second;
    }
    for (int i = 0; i < numQue; ++i) {
        tmp[tmpSize++] = &ques[i].first;
        tmp[tmpSize++] = &ques[i].second;
    }
    sort(tmp,tmp+tmpSize,[&](int*x, int*y) {
        return *x < *y;
    });
    int num = 2;
    int last = *tmp[0];
    *tmp[0] = num;
    for (int i = 1; i < tmpSize; ++i) {
        if (i==0 || last != *tmp[i]) {
            num += 1 + (*tmp[i]-last > 1);
            last = *tmp[i];
        }
        *tmp[i] = num;
    }
    return num;
}

int main() {
    cin.tie(nullptr)->sync_with_stdio(false);
    cin >> numSeg;
    for (int i = 0; i < numSeg; ++i) {
        cin >> segs[i].first >> segs[i].second;
    }   
    cin >> numQue;
    for (int i = 0; i < numQue; ++i) {
        cin >> ques[i].first >> ques[i].second;
    }
    numNode = compress() + 1;
    sort(segs,segs+numSeg);
    for (int i = 2, j = 0, mx = -1; i <= numNode; ++i) {
        while (j < numSeg && segs[j].first <= i) {
            mx = max(mx, segs[j].second);
            j++;
        }
        int x = mx < i ? 1 : mx+1;
        adj[x].push_back(i);
    }
    dfs(1);
    for (int i = 0; i < numQue; ++i) {
        int lef = ques[i].first;
        int rig = ques[i].second;
        int cnt = 0;
        for (int j = LOG-1; j >= 0; --j) {
            if (1 < anc[lef][j] && anc[lef][j] < rig+1) {
                cnt |= 1<<j;
                lef = anc[lef][j];
            }
        }
        if (anc[lef][0] > rig) {
            cout << cnt+1 << '\n';
        }
        else {
            cout << -1 << '\n';
        }
    }
}