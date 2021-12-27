---
title: Z algorithm
category: xu-li-xau
keywords: xau, xâu, string, ham, hàm, function, Z algorithm, Z funcion, mang tien to, mảng tiền tố, prefix array, hai con tro, hai con trỏ, two pointers, linear, tuyen tinh, tuyến tính
---

```cpp
vector<int> Zfunc(const string &s) {
    int n = s.size();
    vector<int> z(n);
    int x = 0, y = 0;
    for (int i = 1; i < n; ++i) {
        z[i] = max(0, min(z[i-x], y-i+1));
        while (i+z[i] < n && s[z[i]] == s[i+z[i]]) {
            x = i; y = i+z[i]; z[i]++;
        }
    }
    return z;
}
```

* Độ phức tạp: $$O(N)$$