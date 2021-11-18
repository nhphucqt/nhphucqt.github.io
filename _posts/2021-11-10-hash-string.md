---
title: Hash String
category: xu-li-xau
keywords: xau, xâu, hash, string, bam, băm, he co so, hệ cơ số, base, khoa, khóa, MOD, modulo, thuat toan khong chuan, thuật toán không chuẩn, xac suat sai, xác suất sai, cau truc du lieu, cấu trúc dữ liệu, data structure
---

<div class="table-of-contents" markdown="1">
* [Hash một khóa](#hash-mot-khoa)
* [Hash nhiều khóa](#hash-nhieu-khoa)
</div>

## Hash một khóa {#hash-mot-khoa}

```cpp
struct Hash {
    const long long MOD = 1e9+7;
    const long long BASE = 311;
    vector<long long> h, p;
    void init(const string &s) {
        int n = s.size();
        h.assign(n+1, 0);
        p.assign(n+1, 1);
        for (int i = 1; i <= n; ++i) {
            p[i] = p[i-1] * BASE % MOD;
            h[i] = (h[i-1] * BASE + s[i-1]) % MOD;
        }
    }
    int getHash(int l, int r) {
        l++; r++;
        return (h[r] - h[l-1] * p[r-l+1] + MOD*MOD) % MOD;
    }
};
```

* Độ phức tạp:
    * Tiền xử lí: $$O(N)$$
    * So sánh xâu: $$O(1)$$
* **Nhận xét:** tốc độ chạy nhanh, có thể đưa ra kết quả sai (vì có mod), để tăng độ chính xác có thể thêm nhiều khóa.

## Hash nhiều khóa {#hash-nhieu-khoa}

```cpp
const int NUMMOD = 4;
const long long MOD[] = {(int)1e9+2277, (int)1e9+5277, (int)1e9+8277, (int)1e9+9277};
const long long BASE = 311;

struct Hash {
   vector<long long> h[NUMMOD], p[NUMMOD];
   void init(const string &s) {
       int n = s.size();
       for (int i = 0; i < NUMMOD; ++i) {
           h[i].resize(n+1);
           p[i].resize(n+1);
           p[i][0] = 1;
           h[i][0] = 0;
           for (int j = 1; j <= n; ++j) {
               h[i][j] = (h[i][j-1] * BASE + s[j-1]) % MOD[i];
               p[i][j] = p[i][j-1] * BASE % MOD[i];
           }
       }
   }
   int getHashWithMod(int l, int r, int i) {
       l++; r++;
       return (h[i][r] - h[i][l-1] * p[i][r-l+1] + MOD[i]*MOD[i]) % MOD[i];
   }
   vector<int> getHash(int l, int r) {
       vector<int> gh;
       for (int i = 0; i < NUMMOD; ++i) {
           gh.push_back(getHashWithMod(l, r, i));
       }
       return gh;
   }
};
```

* **Nhận xét:** có thể tùy chỉnh số lượng khóa (1 đến 4 khóa), có thể đặt giá trị `BASE` ngẫu nhiên để chống hack, bù lại tốc độ chạy sẽ chậm hơn cách [**Hash một khóa**](#hash-mot-khoa).