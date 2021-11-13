#include <bits/stdc++.h>
 
using namespace std;
 
struct Query {
    char k;
    int x, y;
};
 
int n, q;
vector<int> a;
vector<int*> tmp;
vector<Query> que;
 
struct BIT {
    int size;
    vector<int> f, a;
    void init(int n) {
        size = n;
        f.assign(size+1,0);
        a.assign(size+1,0);
    }
    void add(int i, int k) {
        a[i] += k;
        for (int j = i; j <= size; j+=j&-j) {
            f[j] += k;
        }
    }
    int num(int x, int y) {
        int s = 0;
        while (y>=x) {
            if (y-(y&-y)+1 >= x) {
                s += f[y];
                y -= y&-y;
            } 
            else {
                s += a[y];
                y--;
            }
        } return s;
    }
} f;

int main() {
    ios_base::sync_with_stdio(0); cin.tie(0); cout.tie(0);
    cin >> n >> q;
    a.resize(n);
    for (int i = 0; i < n; ++i) {
        cin >> a[i];
        tmp.push_back(&a[i]);
    }
    for (int i = 0; i < q; ++i) {
        char k;
        int x, y;
        cin >> k >> x >> y;
        que.push_back({k,x,y});
    }
    for (int i = 0; i < q; ++i) {
        if (que[i].k=='?') tmp.push_back(&que[i].x);
        tmp.push_back(&que[i].y);
    }
    sort(tmp.begin(),tmp.end(),[&](int*a,int*b) {
        return *a < *b;
    });
    int num = 0;
    for (int i=0,last=-1; i < tmp.size(); ++i) {
        if (i==0 || *tmp[i] != last) {
            last = *tmp[i];
            num++;
        }
        *tmp[i] = num;
    }
    f.init(num);
    for (int i : a) f.add(i,1);
    for (auto& q : que) {
        if (q.k == '!') {
            f.add(a[q.x-1],-1);
            f.add(q.y,1);
            a[q.x-1]=q.y;
        }
        else cout << f.num(q.x,q.y) << '\n';
    }
    return 0;
}