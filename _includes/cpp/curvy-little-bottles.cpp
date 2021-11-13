#include <bits/stdc++.h>

using namespace std;

const int STEP = 1e3;
const double eps = 1e-6;
double W, D, A, K;

struct Func {
    int k;
    vector<double> a, b;
    void init(int k_) {
        k = k_;
        a.resize(k+1);
        b.resize(k+1);
    }
    friend istream& operator >> (istream&is,Func&f) {
        for (int i = 0; i <= f.k; ++i) {
            is >> f.a[i];
        }
        for (int i = 0; i <= f.k; ++i) {
            is >> f.b[i];
        }
        return is;
    }
    Func operator - (double t) {
        Func tmp = *this;
        for (int i = 0; i <= k; ++i) {
            tmp.a[i] -= t*tmp.b[i];
        } return tmp;
    }
    double eval(double x) {
        double A = 0;
        double B = 0;
        for (int i = 0; i <= k; ++i) {
            A += a[i]*pow(x,i);
            B += b[i]*pow(x,i);
        } return A/B;
    }
    double integralPos(double x1, double x2) {
        double d = (x2-x1) / STEP;
        double S = 0;
        for (int i = 0; i < STEP; ++i) {
            double s = (eval(x1+i*d) + eval(x1+(i+1)*d)) * d / 2;
            if (s > 0) S += s;
        } return S;
    }
} yy1, yy2;

double solve() {
    double up = 0;
    double down = -D;
    while (fabs(up-down)>eps) {
        double m = (up+down)/2;
        double S = (yy1-m).integralPos(0,W)-(yy2-m).integralPos(0,W);
        if (S+eps<A) up = m;
        else down = m;
    }
    return -up;
}

int main() {
    ios_base::sync_with_stdio(0); cin.tie(0); cout.tie(0);
    cout << setprecision(5) << fixed;
    while (cin >> W >> D >> A >> K) {
        yy1.init(K); yy2.init(K);
        cin >> yy1 >> yy2;
        cout << solve() << '\n';
    }
    return 0;
}