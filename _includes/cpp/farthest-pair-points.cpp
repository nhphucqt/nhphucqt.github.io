#include <bits/stdc++.h>

using namespace std;

#define Point Vector
struct Vector {
    int x, y;
    Vector() {}
    Vector(int _x, int _y) {
        x = _x; y = _y;
    }
    Vector operator - (const Vector &v) const {
        return Vector(x - v.x, y - v.y);
    }
    long long sqrLen() const {
        return 1LL * x * x + 1LL * y * y;
    }
    long long operator % (const Vector &v) const {
        return 1LL * x * v.y - 1LL * y * v.x;
    }
    friend long long cross (const Vector &a, const Vector &b, const Vector &c) {
        return (b-a) % (c-b);
    }
    friend bool CCW(const Point &a, const Point &b, const Point &c) {
        return cross(a, b, c) > 0;
    }
    friend bool CW(const Point &a, const Point &b, const Point &c) {
        return cross(a, b, c) < 0;
    }
    friend long long sqrDist(const Point &a, const Point &b) {
        return (a-b).sqrLen();
    }
    bool operator < (const Vector &v) const {
        return make_pair(x, y) < make_pair(v.x, v.y);
    }
    bool operator == (const Vector &v) const {
        return make_pair(x, y) == make_pair(v.x, v.y);
    }
};

int numPoint;
vector<Point> points;

vector<Point> ConvexHull(vector<Point> points) {
    sort(points.begin(),points.end());
    points.resize(unique(points.begin(),points.end()) - points.begin());
    if ((int)points.size() <= 2) return points;
    Point lef = points.front();
    Point rig = points.back();
    vector<Point> up, down;
    for (int i = 0; i < (int)points.size(); ++i) {
        if (i == 0 || i == (int)points.size()-1 || CW(lef, points[i], rig)) {
            while ((int)up.size() > 1 && !CW(up[(int)up.size()-2], up.back(), points[i]))
                up.pop_back();
            up.push_back(points[i]);
        }
        if (i == 0 || i == (int)points.size()-1 || CCW(lef, points[i], rig)) {
            while ((int)down.size() > 1 && !CCW(down[(int)down.size()-2], down.back(), points[i]))
                down.pop_back();
            down.push_back(points[i]);
        }
    }
    for (int i = (int)up.size()-2; i > 0; --i) {
        down.push_back(up[i]);
    }
    return down;
}

long long maximumDistance(vector<Point> points) {
    vector<Point> hull = ConvexHull(points);
    long long res = 0;
    for (int i = 0, j = 0; i < (int)hull.size(); ++i) {
        while (j < (int)hull.size()-1 && sqrDist(hull[i], hull[j]) < sqrDist(hull[i], hull[j+1])) {
            j++;
        }
        res = max(res, sqrDist(hull[i], hull[j]));
    }
    return res;
}

int main() {
    cin.tie(nullptr)->sync_with_stdio(false);
    cin >> numPoint;
    for (int i = 0; i < numPoint; ++i) {
        int x, y;
        cin >> x >> y;
        points.emplace_back(x, y);
    }   
    cout << maximumDistance(points);
    return 0;
}