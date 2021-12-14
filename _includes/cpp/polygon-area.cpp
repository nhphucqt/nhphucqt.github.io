#include <bits/stdc++.h>
#define double long double
#define Point Vector
 
using namespace std;
 
struct Vector {
    double x, y;
    Vector() { x = y = 0; }
    Vector(double xx, double yy) {
        x = xx; y = yy;
    }
    Vector operator - (const Vector &v) const {
        return Vector(x-v.x,y-v.y);
    }
    double operator % (const Vector &v)  const {
        return x * v.y - y * v.x;
    }
    friend istream& operator >> (istream&is, Vector&v) {
        is >> v.x >> v.y;
        return is;
    }
};
 
int n;
vector<Point> p;

double polygonArea(const vector<Point> &polygon) {
    double area = polygon.back() % polygon.front();
    for (int i = 0; i+1 < polygon.size(); ++i) {
        area += polygon[i] % polygon[i+1];
    }
    return fabs(area) / 2;
}
 
int main() {
    cin.tie(nullptr)->sync_with_stdio(false);
    cin >> n;
    p.resize(n);
    for (int i = 0; i < n; ++i) {
        cin >> p[i];
    }
    cout << setprecision(6) << fixed;
    cout << polygonArea(p);
    return 0;
}