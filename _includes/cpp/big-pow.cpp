#include <bits/stdc++.h>

using namespace std;

int modulo(const string &s, int MOD) {
    int res = 0;
    for (int i = 0; i < s.size(); ++i) {
        res = (10LL*res%MOD + (s[i]-'0')) % MOD;
    }
    return res;
}

int Pow(int a, int n, int MOD) {
    int res = 1;
    for (;n;n>>=1,a=1LL*a*a%MOD)
        if (n&1) res = 1LL*res*a%MOD;
    return res;
}

int Pow_s(string &sa, string &b, int MOD) {
    int res = 1;
    int a = modulo(sa, MOD);
    for (int i = 0; i < b.size(); ++i) {
        res = 1ll * Pow(res,10,MOD) * Pow(a,b[i]-'0',MOD) % MOD;
    }
    return res;
}

int main() {
    string a, b;
    int MOD;
    cin >> a >> b;
    cin >> MOD;
    cout << Pow_s(a, b, MOD);
    return 0;
}