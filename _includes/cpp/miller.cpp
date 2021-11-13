//                   n                   |                      a                         |
// n < 2,047                             | 2                                              |
// n < 1,373,653                         | 2, 3                                           |
// n < 9,080,191                         | 31, 73                                         |
// n < 4,759,123,141                     | 2, 7, 61                                       |
// n < 1,122,004,669,633                 | 2, 13 , 23, 1662803                            |
// n < 2,152,302,898,747                 | 2, 3, 5, 7, 11                                 |
// n < 3,474,749,660,383                 | 2, 3, 5, 7, 11, 13                             |
// n < 341,550,071,728,321               | 2, 3, 5, 7, 11, 13, 17                         |
// n < 3,825,123,056,546,413,051         | 2, 3, 5, 7, 11, 13, 17, 19, 23                 |
// n < 318,665,857,834,031,151,167,461   | 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37     |
// n < 3,317,044,064,679,887,385,961,981 | 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41 |

#include <bits/stdc++.h>

using namespace std;

typedef long long ll;

ll mul(ll a, ll b, ll MOD) { // if n ~ 64bit
    ll res = 0;
    a %= MOD;
    b %= MOD;
    while (b>0) {
        if (b&1) res = (res + a) % MOD;
        a = a*2%MOD; b >>= 1;
    }
    return res;
}

ll Pow(ll a, ll n, ll MOD) {
    ll res = 1;
    a %= MOD;
    for (;n;n>>=1,a=a*a%MOD)
        if (n&1) res = res * a % MOD;
    return res;
}

bool check(ll n, ll d, ll s, ll a) {
//    if (n == a) return true; // use when not using loops
    ll p = Pow(a, d, n);
    if (p == 1) return true;
    while (s--) {
        if (p == n-1) return true;
        p = p*p%n;
    }
    return false;
}

bool prime(ll n) {
    if (n < 2) return false;
    if (!(n&1)) return n == 2;
    ll d = n-1, s = 0;
    while (!(d&1)) { s++; d >>= 1; }
//    return check(n, d, s, 2) && check(n, d, s, 3); // ~ 10^6
//    return check(n, d, s, 2) && check(n, d, s, 7) && check(n, d, s, 61); // 32bit
    for (ll a = 2; a <= min(n-1, ll(2*log(n)*log(n))); ++a)
        if (!check(n, d, s, a)) return false;
    return true;
}

int main() {
    ll n;
    cin >> n;
    if (prime(n)) cout << "YES";
    else cout << "NO";
    return 0;
}