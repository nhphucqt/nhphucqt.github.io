#include <bits/stdc++.h>

using namespace std;

#define Point Vector
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
    long long sqrLen() const {
        return 1LL * x * x + 1LL * y * y;
    }
    friend istream& operator >> (istream&is, Vector &v) {
        is >> v.x >> v.y;
        return is;
    }
    friend ostream& operator << (ostream&os, const Vector &v) {
        os << v.x << ' ' << v.y;
        return os;
    }
};

int numPoint;
vector<Point> points;

int main() {
    cin.tie(nullptr)->sync_with_stdio(false);
    cin >> numPoint;
    points.resize(numPoint);
    for (int i = 0; i < numPoint; ++i) {
        cin >> points[i];
    }
    Point pivot = points[0];
    for (int i = 1; i < numPoint; ++i) {
        if (make_pair(pivot.y,pivot.x) > make_pair(points[i].y,points[i].x)) {
            pivot = points[i];
        }
    }
    sort(points.begin(),points.end(),[&](const Point &a, const Point &b) {
        Point u = a - pivot;
        Point v = b - pivot;
        long long tmp = u % v;
        if (tmp != 0) return tmp > 0;
        return u.sqrLen() < v.sqrLen();
    });
    for (int i = 0; i < numPoint; ++i) {
        cout << points[i] << '\n';
    }
    return 0;
}