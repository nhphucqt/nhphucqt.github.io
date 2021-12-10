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
    friend istream& operator >> (istream&is, Vector &v) {
        is >> v.x >> v.y;
        return is;
    }
    friend ostream& operator << (ostream&os, const Vector &v) {
        os << v.x << ' ' << v.y;
        return os;
    }
    bool operator < (const Vector &v) const {
        return make_pair(x,y) < make_pair(v.x,v.y);
    }
    friend bool CCW(const Point &a, const Point &b, const Point &c) {
        return (b-a) % (c-b) > 0;
    }
    friend bool CW(const Point &a, const Point &b, const Point &c) {
        return (b-a) % (c-b) < 0;
    }
};

int numPoint;
vector<Point> points;

vector<Point> ConvexHull(vector<Point> points) {
    if (points.size() <= 2) return points;
    sort(points.begin(),points.end());
    vector<Point> up, down;
    Point lef = points[0];
    Point rig = points.back();
    for (int i = 0; i < points.size(); ++i) {
        if (!CCW(lef, points[i], rig)) {
            while (up.size() > 1 && CCW(up[(int)up.size()-2], up.back(), points[i])) {
                up.pop_back();
            }
            up.push_back(points[i]);
        }
        if (!CW(lef, points[i], rig)) {
            while (down.size() > 1 && CW(down[(int)down.size()-2], down.back(), points[i])) {
                down.pop_back();
            }
            down.push_back(points[i]);
        }
    }
    // Clockwise
    for (int i = (int)down.size()-2; i > 0; --i) {
        up.push_back(down[i]);
    }
    return up;
}

int main() {
    cin.tie(nullptr)->sync_with_stdio(false);
    cin >> numPoint;
    points.resize(numPoint);
    for (int i = 0; i < numPoint; ++i) {
        cin >> points[i];
    }
    vector<Point> hull = ConvexHull(points);
    cout << hull.size() << '\n';
    for (int i = 0; i < hull.size(); ++i) {
        cout << hull[i] << '\n';
    }
    return 0;
}