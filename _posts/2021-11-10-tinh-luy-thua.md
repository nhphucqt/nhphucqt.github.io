---
title: Tính lũy thừa 
category: so-hoc
keywords: so hoc, số học, number theory, luy thua, lũy thừa, pow, power, chia de tri, chia để trị, divide and conquer, khu de quy, khử đệ quy, so lon, số lớn, big number, bignum
---

<div class="table-of-contents" markdown="1">
* [Độ phức tạp $$O(N)$$ - cơ bản](#co-ban)
* [Độ phức tạp $$O(log(N))$$ - chia để trị](#chia-de-tri)
* [Độ phức tạp $$O(log(N))$$ - khử đệ quy](#khu-de-quy)
* [Lũy thừa hai cấp](#luy-thua-hai-cap)
* [Lũy thừa hai số lớn](#luy-thua-hai-so-lon)
</div>

#### Bài viết này sẽ tập trung vào cách tính lũy thừa $$\large a^n \bmod M$$

## Độ phức tạp $$O(N)$$ - cơ bản {#co-ban}

```cpp
int Pow(int a, int n, int M) {
    int res = 1;
    while (n--) res = 1LL*res*a%M;
    return res;
}
```

## Độ phức tạp $$O(log(N))$$ - Chia để trị {#chia-de-tri}
Áp dụng công thức: $$\large a^n = a^{\lfloor\frac{n}{2}\rfloor} \times a^{\lfloor\frac{n}{2}\rfloor} \times a^{n \hspace{1mm}\bmod\hspace{1mm} 2}$$

```cpp
int Pow(int a, long long n, int M) {
    if (n == 0) return 1;
    int tmp = Pow(a, n/2, M);
    tmp = 1LL * tmp * tmp % M;
    if (n&1) tmp = 1LL * tmp * a % M;
    return tmp;
}
```

## Độ phức tạp $$O(log(N))$$ - Khử đệ quy {#khu-de-quy}
Dùng đệ quy để tính lũy thừa không phải lúc nào cũng hiệu quả, nhất là trong nhân ma trận (nếu kích cỡ ma trận lớn thì dễ bị tràn stack), thay vào đó chúng ta có thể dùng cách khử đệ quy.

**Ưu điểm:**
* Chạy nhanh hơn;
* Tốn ít bộ nhớ hơn, tránh tràn stack.

Thuật toán sẽ khác hoàn toàn với với cách chia để trị, để tính $$a^n$$ ta sẽ làm như sau:
* Chuyển $$n$$ thành dãy nhị phân, như vậy $$n = 2^{x_1} + 2^{x_2} + 2^{x_3} + \ldots + 2^{x_k}$$, khi đó:

$$\large a^n = a^{2^{x_1} + 2^{x_2} + 2^{x_3} + \ldots + 2^{x_k}} = a^{2^{x_1}} \times a^{2^{x_2}} \times a^{2^{x_3}} \times \ldots \times a^{2^{x_k}}$$

```cpp
int Pow(int a, long long n, int M) {
    int res = 1;
    for (;n;n>>=1,a=1LL*a*a%M) 
        if (n&1) res = 1LL*res*a%M;
    return res;
}
```

## Lũy thừa hai cấp {#luy-thua-hai-cap}
**Bài toán:** cho ba số $$a$$, $$b$$, $$c$$ $$(0 \le a, b, c \le 10^9)$$, tính $$a^{b^c} \bmod (10^9+7)$$

**Ý tưởng**
* **Định lý Fermat nhỏ:** $$x^{p-1} \equiv 1 \pmod p$$ với $$x$$ là số tự nhiên, $$p$$ là số nguyên tố và $$(x, p) = 1$$
* Đặt $$p = 10^9+7$$, ta có $$b^c = k \times (q-1) + r$$ (tức $$b^c / (p-1) = k$$ dư $$r$$)
* Khi đó $$a^{b^c} = a^{k \times p+r} = a^{k \times (p-1)} \times a^r = (a^{p-1})^k \times a^r$$
* Vì $$a$$ và $$p$$ nguyên tố cùng nhau (do $$a < p$$), áp dụng định lý Fermat nhỏ: $$a^{p-1} \equiv 1 \pmod p \Rightarrow (a^{p-1})^k \equiv 1 \pmod p$$
* Vậy nên $$a^{p-1} \times a^r \equiv a^r \pmod p \Rightarrow a^{b^c} \equiv a^{b^c \hspace{1mm}\bmod\hspace{1mm} (p-1)} \bmod p$$

**Kết quả của bài toán là:** Pow(a, Pow(b, c, p-1), p)

## Lũy thừa hai số lớn {#luy-thua-hai-so-lon}
Tính $$a^b \bmod MOD$$, với $$a, b$$ là 2 số nguyên dương nhỏ $$\le 10^{100000}, MOD \le 10^9$$

**Ý tưởng:** [codeforces.com/blog/entry/60509#comment-443755](https://codeforces.com/blog/entry/60509#comment-443755)

```cpp
{% include cpp/big-pow.cpp %}
```