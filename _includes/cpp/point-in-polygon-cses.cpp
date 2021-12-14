#include <bits/stdc++.h>
#define Point Vector

using namespace std;

const double PI = acos(-1);
const double EPS = 1e-9;

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
    long long operator * (const Vector &v) const {
        return 1LL * x * v.x + 1LL * y * v.y;
    }
    friend bool collinear(const Point &a, const Point &b, const Point &c) {
        return (b-a) % (c-b) == 0;
    }
    friend double angle(const Vector &a, const Vector &b) {
        return atan2(a%b,a*b);
    }
    friend istream& operator >> (istream&is, Vector &v) {
        is >> v.x >> v.y;
        return is;
    }
};

const int N = 5007;
int polySize, numPoint;
Point polygon[N];

bool isBoundary(const Point &p) {
    for (int i = 0; i < polySize; ++i) {
        if (collinear(polygon[i],polygon[i+1],p)) {
            if ((polygon[i]-p) * (polygon[i+1]-p) <= 0) 
                return true;
        }
    }
    return false;
}

bool isInside(const Point &p) { // not isBoundary
    double sum = 0;
    for (int i = 0; i < polySize; ++i) {
        sum += angle(polygon[i]-p,polygon[i+1]-p);
    }
    return fabs(fabs(sum) - PI*2) < EPS;
}

int main() {
    cin.tie(nullptr)->sync_with_stdio(false);
    cin >> polySize >> numPoint;
    for (int i = 0; i < polySize; ++i) {
        cin >> polygon[i];
    }
    polygon[polySize] = polygon[0];
    while (numPoint--) {
        Point p; cin >> p;
        if (isBoundary(p)) {
            cout << "BOUNDARY\n";
        }
        else if (isInside(p)) {
            cout << "INSIDE\n";
        }
        else {
            cout << "OUTSIDE\n";
        }
    }
    return 0;
}