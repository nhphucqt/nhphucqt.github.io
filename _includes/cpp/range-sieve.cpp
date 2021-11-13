#include <bits/stdc++.h>

using namespace std;

const int N = 1e7+7;
bool isPrime[N];

void sieve(long long l, long long r) {
    memset(isPrime, true, sizeof isPrime);
    if (l == 0) isPrime[0] = isPrime[1] = false;
    else if (l == 1) isPrime[0] = false;
    for (long long i = 2; i*i <= r; ++i) 
    for (long long j = max(i*2,l+(i-l%i)%i); j <= r; j+=i)
        isPrime[j-l] = false;
}

int main() {
    long long l, r;
    cin >> l >> r;
    sieve(l, r);
    for (int i = l; i <= r; ++i) {
        cout << i << ' ' << isPrime[i-l] << '\n';
    }
    return 0;
}