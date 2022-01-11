---
title: Tìm khoảng cách lớn nhất, nhỏ nhất giữa hai cặp điểm
category: hinh-hoc
keywords: hinh hoc, hình học, geometry, cap diem, cặp điểm, gan nhat, gần nhất, xa nhat, xa nhất, pair points, closest, farthest, bao loi, bao lồi, convex hull, duong quet, đường quét, sweep line
---

<div class="table-of-contents" markdown="1">
* [Tìm khoảng cách lớn nhất giữa hai cặp điểm](#khoang-cach-lon-nhat)
* [Tìm khoảng cách nhỏ nhất giữa hai cặp điểm](#khoang-cach-nho-nhat)
* [Nguồn tham khảo](#tham-khao)
</div>

## Tìm khoảng cách lớn nhất giữa hai cặp điểm {#khoang-cach-lon-nhat}

**Bài toán:** Trên mặt phẳng tọa độ cho $$N$$ điểm tọa độ nguyên $$(X_i, Y_i)$$ $$(2 \le N \le 10^5; \lvert X_i \rvert, \lvert Y_i \rvert \le 10^9)$$, tìm bình phương khoảng cách lớn nhất của hai cặp điểm bất kì trong tập điểm đã cho.

**Độ phức tạp:** $$O(Nlog(N))$$

```cpp
{% include cpp/farthest-pair-points.cpp %}
```

## Tìm khoảng cách nhỏ nhất giữa hai cặp điểm {#khoang-cach-nho-nhat}

**Đề bài:** [https://cses.fi/problemset/task/2194](https://cses.fi/problemset/task/2194)

**Độ phức tạp:** $$O(Nlog(N))$$

```cpp
{% include cpp/closest-pair-points.cpp %}
```

## Nguồn tham khảo {#tham-khao}

* [https://www.topcoder.com/thrive/articles/Line Sweep Algorithms](https://www.topcoder.com/thrive/articles/Line Sweep Algorithms)
* [https://codeforces.com/blog/entry/58747](https://codeforces.com/blog/entry/58747)