#include <bits/stdc++.h>

using namespace std;

template<typename T>
inline void read(T &x) {
    bool neg = false;
    char c;
    for (c = getchar(); !isdigit(c); c = getchar())
        if (c == '-') neg = !neg;
    x = c-'0';
    for (c = getchar(); isdigit(c); c = getchar())
        x = x*10 + (c-'0');
    if (neg) x = -x;
}

template<typename T>
inline void write(T x) {
    if (x < 0) { putchar('-'); x = -x; }
    T tmp = x/10, de = 1;
    while (tmp > 0) { de *= 10; tmp /= 10; }
    while (de > 0) { putchar(x/de+'0'); x %= de; de /= 10; }
}

template<typename T>
inline void writeln(T x) {
    write(x);
    putchar('\n');
}

const int N = 1e7+7;
int n, a[N];

int main() {
    read(n);
    for (int i = 1; i <= n; ++i) {
        read(a[i]);
        writeln(a[i]);
    }
    return 0;
}