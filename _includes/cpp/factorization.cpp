#include <bits/stdc++.h>

using namespace std;

vector<pair<long long,int>> factorize(long long n) {
    vector<pair<long long, int>> fac;
    for (long long i = 2; i*i <= n; ++i) {
        if (n % i == 0) {
            fac.emplace_back(i, 0);
            while (n % i == 0) {
                fac.back().second++;
                n /= i;
            }
        }
    }
    if (n > 1) fac.emplace_back(n, 1);
    return fac;
}

int main() {
    long long n; cin >> n;
    auto fac = factorize(n);
    for (auto f : fac) cout << f.first << ' ' << f.second << '\n';
    return 0;
}