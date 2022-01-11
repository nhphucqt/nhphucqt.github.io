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
        return Vector(x-v.x,y-v.y);
    }
    long long sqrLen() const {
        return 1LL * x * x + 1LL * y * y;
    }
    friend long long sqrDist(const Point &a, const Point &b) {
        return (a-b).sqrLen();
    }
    bool operator < (const Vector &v) const {
        return make_pair(y, x) < make_pair(v.y, v.x);
    }
};

const long long INFLL = 9e18;
const int INF = 2e9;
int numPoint;
vector<Point> points;

long long mySqr(long long x) {
    return x * x;
}

long long ceilSqrt(long long x) {
    long long i = sqrt(x);
    while (i*i > x) i--;
    while (i*i < x) i++;
    return i;
}

long long minimumDistance(vector<Point> points) {
    set<Point> heap;
    long long cur = INFLL;
    sort(points.begin(), points.end(), [&](const Point &a, const Point &b) {
        return make_pair(a.x, a.y) < make_pair(b.x, b.y);
    });
    for (int i = 0, j = 0; i < (int)points.size(); ++i) {
        while (mySqr(points[i].x-points[j].x) >= cur) {
            heap.erase(points[j]);
            j++;
        }
        if (!heap.empty()) {
            long long tmp = ceilSqrt(cur);
            auto down = heap.upper_bound(Point(points[i].x, max(-1LL * INF, points[i].y - tmp)));
            auto up = heap.upper_bound(Point(points[i].x, min(1LL * INF, points[i].y + tmp)));
            for (auto it = down; it != up; ++it) {
                if (sqrDist(points[i], *it) < cur) {
                    cur = sqrDist(points[i], *it);
                }
            }
        }
        heap.insert(points[i]);
    }
    return cur;
}

int main() {
    cin.tie(nullptr)->sync_with_stdio(false);
    cin >> numPoint;
    for (int i = 0; i < numPoint; ++i) {
        int x, y;
        cin >> x >> y;
        points.emplace_back(x, y);
    }   
    cout << minimumDistance(points);
    return 0;
}