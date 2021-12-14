#include <bits/stdc++.h>
#define Point Vector

using namespace std;

struct Vector {
    int x, y;
    Vector() { x = y = 0; }
    Vector(int xx, int yy) {
        x = xx; y = yy;
    }
    Vector operator - (const Vector &v) const {
        return Vector(x-v.x,y-v.y);
    }
    long long operator % (const Vector &v) const {
        return 1LL * x * v.y - 1LL * y * v.x;
    }
    friend bool CCW(const Point &a, const Point &b, const Point &c) {
        return (b-a) % (c-b) > 0;
    }
    friend istream& operator >> (istream&is, Vector &v) {
        is >> v.x >> v.y;
        return is;
    }
};

const int N = 5007;
int polySize, numPoint;
Point polygon[N];

bool inPolygon(const Point &p) {
    for (int i = 0; i < polySize; ++i) {
        if (!CCW(polygon[i],polygon[i+1],p))
            return false;
    }
    return true;
}

int main() {
    cin.tie(nullptr)->sync_with_stdio(false);
    cin >> polySize;
    for (int i = 0; i < polySize; ++i) {
        cin >> polygon[i];
    }
    polygon[polySize] = polygon[0];
    cin >> numPoint;
    while (numPoint--) {
        Point p; cin >> p;
        cout << (inPolygon(p) ? "YES\n" : "NO\n");
    }
    return 0;
}