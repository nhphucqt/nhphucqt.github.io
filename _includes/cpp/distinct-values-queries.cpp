#include <bits/stdc++.h>

using namespace std;

int block;
struct Query {
   int l, r, idx;
   pair<int,int> toPair() const {
       return make_pair(l/block, l/block & 1 ? -r : r);
   }
};

const int N = 2e5+7;
int n, q, a[N];
vector<Query> que;
int cnt[N], res, ans[N];

void compress() {
   int *tmp[n];
   for (int i = 1; i <= n; ++i) tmp[i-1] = &a[i];
   sort(tmp,tmp+n,[&](int*x,int*y) { return *x < *y; });
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
   compress();
   block = sqrt(n);
   sort(que.begin(),que.end(),[&](const Query& a, const Query& b) {
       return a.toPair() < b.toPair();
   });
   int l = 1, r = 0;
   for (int i = 0; i < q; ++i) {
       for (; l < que[i].l; res-=--cnt[a[l]]==0, ++l);
       for (; l > que[i].l; --l, res+=++cnt[a[l]]==1);
       for (; r > que[i].r; res-=--cnt[a[r]]==0, --r);
       for (; r < que[i].r; ++r, res+=++cnt[a[r]]==1);
       ans[que[i].idx] = res;
   }
   for (int i = 0; i < q; ++i) cout << ans[i] << '\n';
   return 0;
}