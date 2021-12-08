---
title: Hàm modulo
category: ki-thuat
keywords: ki thuat, kĩ thuật, ham, hàm, function, so hoc, số học, number theory, mod, modulo, so du, số dư
---

```cpp
const int MOD = 1e9+7;
inline int add(int x, int y) {
    if ((x+=y) >= MOD) x -= MOD;
    return x;
}
inline void selfAdd(int &x, int y) {
    if ((x+=y) >= MOD) x -= MOD;
}
inline int sub(int x, int y) {
    if ((x-=y) < 0) x += MOD;
    return x;
}
inline void selfSub(int &x, int y) {
    if ((x-=y) < 0) x += MOD;
}
inline int mul(long long x, int y) {
    return x * y % MOD;
}
inline void selfMul(int &x, int y) {
    x = 1LL * x * y % MOD;
}
```