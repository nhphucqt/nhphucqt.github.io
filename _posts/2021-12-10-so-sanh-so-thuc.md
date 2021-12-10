---
title: So sánh số thực
category: ki-thuat
keywords: ki thuat, kĩ thuật, so sanh, so sánh, compare, so thuc, số thực, float number, eps, epsilon
---

```cpp
const double EPS = 1e-9;
double x, y;
fabs(x - y) < EPS; // x == y
fabs(x - y) > EPS; // x != y
x + EPS < y; // x < y
x < y + EPS; // x <= y
x > y + EPS; // x > y
x + EPS > y; // x >= y
```