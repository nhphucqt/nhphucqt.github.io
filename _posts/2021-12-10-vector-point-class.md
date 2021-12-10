---
title: Vector/Point class
category: hinh-hoc
keywords: hinh hoc, hình học, geometry, vector, point, cau truc du lieu, cấu trúc dữ liệu, data structure
---

```cpp
const double EPS = 1e-9;
#define Point Vector
struct Vector {
    double x, y;
    Vector() { x = y = 0; }
    Vector(double xx, double yy) {
        x = xx; y = yy;
    }
    Vector operator + (const Vector &v) const {
        return Vector(x+v.x,y+v.y);
    }
    Vector operator - (const Vector &v) const {
        return Vector(x-v.x,y-v.y);
    }
    Vector operator * (double k) const {
        return Vector(x*k,y*k);
    }
    Vector rotate(double a) const { // a rad
        return Vector(x*cos(a) - y*sin(a), y*cos(a) + x*sin(a));
    }
    double operator * (const Vector &v) const {
        return x * v.x + y * v.y;
    }
    double operator % (const Vector &v) const {
        return x * v.y - y * v.x;
    }
    double sqrLen() const {
        return x * x + y * y;
    }
    double len() const {
        return sqrt(sqrLen());
    }
    friend double angle(const Vector &a, const Vector &b) { // rad
        return atan2(a%b,a*b);
    }
    friend double dist(const Point &a, const Point &b) {
        return (a-b).len();
    }
    friend bool collinear(const Point &a, const Point &b, const Point &c) {
        return fabs((b-a) % (c-b)) < EPS;
    }
    friend bool CCW(const Point &a, const Point &b, const Point &c) {
        return ((b-a) % (c-b)) > EPS;
    }
    friend bool CW(const Point &a, const Point &b, const Point &c) {
        return ((b-a) % (c-b)) + EPS < 0;
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
```