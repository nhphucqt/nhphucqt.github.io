#include <bits/stdc++.h>

using namespace std;

struct Query {
    int l, r, idx;
};

struct BIT {
    int n;
    vector<int> f;
    void init(int nn) {
        n = nn;
        f.assign(n+1, 0);
    }
    void add(int i, int k) {
        if (i == 0) return;
        for (; i <= n; i+=i&-i) f[i] += k;
    }
    int sum(int x, int y) {
        int s = 0;
        for (; y; y-=y&-y) s += f[y];
        for (x--; x; x-=x&-x) s -= f[x];
        return s;
    }
} fw;

const int N = 2e5+7;
int n, q, a[N];
vector<Query> que;
int last[N], Prev[N], ans[N];

void compress() {
    int *tmp[n];
    for (int i = 1; i <= n; ++i) tmp[i-1] = &a[i];
    sort(tmp,tmp+n,[&](int*a,int*b) { return *a < *b; });
    for (int i = 0, num = 0, last; i < n; ++i) {
        if (i==0 || last != *tmp[i]) {
            num++; last = *tmp[i];
        } *tmp[i] = num;
    }
}

int main() {
    cin.tie(nullptr)->sync_with_stdio(false);
    cin >> n >> q;
    for (int i = 1; i <= n; ++i) cin >> a[i];
    for (int i = 0; i < q; ++i) {
        int l, r; cin >> l >> r;
        que.push_back({l,r,i});
    }
    sort(que.begin(),que.end(),[&](const Query& a, const Query& b) {
        return a.r < b.r;
    });
    fw.init(n);
    compress();
    for (int i = 1; i <= n; ++i) fw.add(i,1);
    for (int i = 1; i <= n; ++i) {
        Prev[i] = last[a[i]];
        last[a[i]] = i;
    }   
    for (int i = 0, r = 0; i < q; ++i) {
        for (; r < que[i].r; ++r, fw.add(Prev[r],-1));
        ans[que[i].idx] = fw.sum(que[i].l,que[i].r);
    }
    for (int i = 0; i < q; ++i) cout << ans[i] << ' ';
    return 0;
}