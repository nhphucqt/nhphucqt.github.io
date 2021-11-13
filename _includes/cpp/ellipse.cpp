#include <bits/stdc++.h>

using namespace std;

const double eps = 1e-12;
double a, b;

double f(double x) {
    /* function here */
    return b*sqrt((1-x/a)*(1+x/a));
}
double simpson(double x1, double x2) {
    return (x2-x1)/6*(f(x1)+f(x2)+4*f((x1+x2)/2));
}
double integral(double x1, double x2, double ans) {
    double m = (x1+x2)/2;
    double lef = simpson(x1,m);
    double rig = simpson(m,x2);
    if (fabs(lef+rig-ans) < eps) return ans;
    return integral(x1,m,lef) + integral(m,x2,rig);
}
double integral(double x1, double x2) {
    return integral(x1,x2,simpson(x1,x2));
}

int main() {
    int t; cin >> t;
    cout << setprecision(3) << fixed;
    while (t--) {
        double x1, x2;
        cin >> a >> b >> x1 >> x2;
        cout << integral(x1,x2)*2 << '\n';
    }
    return 0;
}