// link de: http://online.vku.udn.vn/problem/joyful

#include <bits/stdc++.h>
#define fi first
#define se second
#define debug(x) cerr << #x << " = " << x << '\n'

using namespace std;

struct Point {
    int x, y;
    friend istream& operator >> (istream&is,Point&p) {
        is >> p.x >> p.y;
        return is;
    }
};

struct Segment {
    Point a, b;
    friend istream& operator >> (istream&is,Segment&s) {
        is >> s.a >> s.b;
        if (s.a.x > s.b.x) swap(s.a.x,s.b.x);
        if (s.a.y > s.b.y) swap(s.a.y,s.b.y);
        return is;
    }
    bool isHor() {
        return a.y == b.y;
    }
};

struct item {
    int x, y, z;
    bool operator < (item&i) {
        return x < i.x;
    }
};

const int N = 2e5+7;
int n;
Segment s[N];
int *B[2*N], maxY, val;

namespace sub2 {
    item hor[N*2], ver[N];
    int shor = 0, sver = 0;
    struct BIT {
        int n;
        int f[2*N], a[2*N];
        void init(int n_) {
            n = n_;
        }
        void update(int i, int k) {
            for (int j = i; j <= n; j+=j&-j) {
                f[j] += k;
            }
            a[i] += k;
        }
        int sum(int x, int y) {
            int res = 0;
            while (y >= x) {
                if (y-(y&-y)+1 >= x) {
                    res += f[y];
                    y -= y&-y;
                }
                else { res += a[y--]; }
            } return res;
        }
    } f;

    long long solve() {
        long long res = 0;
        for (int i = 0; i < n; ++i) {
            if (s[i].isHor()) {
                hor[shor++] = {s[i].a.x,s[i].a.y,1};
                hor[shor++] = {s[i].b.x+1,s[i].b.y,-1};
            }
            else {
                ver[sver++] = {s[i].a.x,s[i].a.y,s[i].b.y};
            }
        }
        sort(hor,hor+shor);
        sort(ver,ver+sver);
        f.init(maxY);
        int j = 0;
        while (j < sver && ver[j].x < hor[0].x) j++;
        for (int i=0; i < shor; ++i) {
            f.update(hor[i].y,hor[i].z);
            if (i+1 == shor) break;
            if (hor[i].x != hor[i+1].x) {
                while (j < sver && ver[j].x < hor[i+1].x) {
                    res += f.sum(ver[j].y,ver[j].z);
                    j++;
                }
            }
            if (j == sver) break;
        }
        return res;
    }
}

int main() {
    ios_base::sync_with_stdio(0); cin.tie(0); cout.tie(0);
    cin >> n;
    for (int i = 0; i < n; ++i) {
        cin >> s[i];
    }
    for (int i = 0; i < n; ++i) {
        B[i] = &s[i].a.y;
        B[i+n] = &s[i].b.y;
    }
    sort(B,B+2*n,[&](int *a, int *b) {
        return *a < *b;
    });
    for (int i = 0; i < 2*n; ++i) {
        if (i == 0 || *B[i] != val) { maxY++; val = *B[i]; }
        *B[i] = maxY;
    }
    cout << sub2::solve();
    return 0;
}