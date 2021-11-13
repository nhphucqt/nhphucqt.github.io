#include <bits/stdc++.h>

using namespace std;

const double eps = 1e-6;
const double PI = acos(-1);

double cub(double x) {
    return x*x*x;
}

double cubr(double x) {
    if (x < 0) return -pow(-x,1.0/3);
    return pow(x,1.0/3);
}

vector<double> FirstDegreeEquation(double a, double b) {
    return {-b/a};   
}

vector<double> QuadraticEquation(double a, double b, double c) {
    double delta = b*b - 4*a*c;
    if (delta < 0) return {};
    if (fabs(delta) < eps) return {-b/(2*a)};
    double x1 = (-b-sqrt(delta)) / (2*a);
    double x2 = (-b+sqrt(delta)) / (2*a);
    return {x1,x2};
}

vector<double> CubicEquation(double a, double b, double c, double d) {
    double delta = b*b - 3*a*c;
    if (abs(delta) < eps) {
        double k = b*b*b - 27*a*a*d;
        if (fabs(k) < eps) return {-b/(3*a)};
        else return {(-b+cubr(k))/(3*a)};
    }
    else {
        double k = (9*a*b*c - 2*b*b*b - 27*a*a*d) / (2*sqrt(cub(fabs(delta))));
        if (delta > 0) {
            if (fabs(k) <= 1) {
                double x1 = (2*sqrt(delta)*cos(acos(k)/3) - b) / (3*a);
                double x2 = (2*sqrt(delta)*cos(acos(k)/3 - 2*PI/3) - b) / (3*a);
                double x3 = (2*sqrt(delta)*cos(acos(k)/3 + 2*PI/3) - b) / (3*a);
                return {x1,x2,x3};
            }
            else {
                double x = sqrt(fabs(delta))*fabs(k)/(3*a*k);
                x *= cubr(fabs(k)+sqrt(k*k-1)) + cubr(fabs(k)-sqrt(k*k-1));
                x -= b / (3*a);
                return {x};
            }
        }
        else {
            double x = sqrt(fabs(delta))/(3*a);
            x *= cubr(k+sqrt(k*k+1)) + cubr(k-sqrt(k*k+1));
            x -= b / (3*a);
            return {x};
        }
    }
}

int main() {
    auto x1 = FirstDegreeEquation(2,3); // 2x + 3 = 0
    auto x2 = QuadraticEquation(3,-2,-1); // 3x^2 - 2x - 1 = 0
    auto x3 = CubicEquation(3,14,-47,14); // 3x^3 + 14x^2 - 47x + 14 = 0

    cout << setprecision(6) << fixed;
    for (double x : x1) cout << x << ' ';
    cout << '\n';
    for (double x : x2) cout << x << ' ';
    cout << '\n';
    for (double x : x3) cout << x << ' ';
    cout << '\n';

    return 0;
}
