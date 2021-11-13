#include <bits/stdc++.h>

using namespace std;

const double eps = 1e-12;
const double PI = acos(-1);
int n;
double a, b;

double f(double x) {
    return b*b*(1-x/a)*(1+x/a);
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
    ios_base::sync_with_stdio(0); cin.tie(0); cout.tie(0);
    cout << setprecision(6) << fixed;
    cin >> b >> a >> n;
    double d = a / n;
    a /= 2; b /= 2;
    for (int i = 0; i < n; ++i) {
        cout << integral(-a+i*d,-a+(i+1)*d)*PI << '\n';
    }
    return 0;
}