---
title: Kiểm tra điểm nằm trong đa giác
category: hinh-hoc
keywords: hinh hoc, hình học, geometry, point, polygon, point in polygon, diem, điểm, da giac, đa giác, tich cheo, tích chéo, cross product, goc, góc, angle
---

<div class="table-of-contents" markdown="1">
* [Trường hợp kiểm tra trong đa giác lồi](#da-giac-loi)
    * [Duyệt từng cạnh - $$O(N)$$](#duyet-tung-canh)
    * [Chặt nhị phân - $$O(log(N))$$](#chat-nhi-phan)
* [Trường hợp kiểm tra trong đa giác tổng quát](#da-giac-tong-quat)
</div>

## Trường hợp kiểm tra trong đa giác lồi {#da-giac-loi}

**Đề bài:** [https://oj.vnoi.info/problem/meterain](https://oj.vnoi.info/problem/meterain)

### Duyệt từng cạnh - $$O(N)$$ {#duyet-tung-canh}

```cpp
{% include cpp/meterain-vnoi.cpp %}
```

### Chặt nhị phân - $$O(log(N))$$ {#chat-nhi-phan}

```cpp
{% include cpp/meterain-binarysearch-vnoi.cpp %}
```

## Trường hợp kiểm tra trong đa giác tổng quát {#da-giac-tong-quat}

**Đề bài:** [https://cses.fi/problemset/task/2192/](https://cses.fi/problemset/task/2192/)

Độ phức tạp: $$O(N)$$

```cpp
{% include cpp/point-in-polygon-cses.cpp %}
```