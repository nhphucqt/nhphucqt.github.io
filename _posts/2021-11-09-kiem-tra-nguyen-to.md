---
title: Kiểm tra nguyên tố
category: so-hoc
---

<div class="table-of-contents" markdown="1">
* [$$O(N)$$ - cơ bản](#co-ban)
* [$$O(\sqrt{N})$$ - Tối ưu](#toi-uu)
* [$$O(Klog(N)) / O(Klog(N)^2)$$ - Miller](#miller)
</div>

## $$O(N)$$ - Cơ bản {#co-ban}
Để kiểm tra một số n có phải là số nguyên tố hay không, ta có thể duyệt $$i$$ từ $$2$$ đến $$n-1$$:
- Nếu tồn tại $$i$$ sao cho $$n$$ chia hết cho $$i$$ thì $$n$$ có nhiều hơn $$2$$ ước dương $$\Rightarrow$$ $$n$$ không phải là số nguyên tố.
- Nếu không tồn tại thì n chỉ có 2 ước dương là 1 và n $$\Rightarrow$$ n là số nguyên tố.
**Lưu ý:** các số nhỏ hơn 2 không phải là hợp số cũng không phải là số nguyên tố.

```cpp
bool isPrime(int n) {
    if (n < 2) return false;
    for (int i = 2; i <= n-1; ++i)
        if (n % i == 0) return false;
    return true;
}
```

## $$O(\sqrt{N})$$ - Tối ưu {#toi-uu}
Có thể dễ dàng chứng minh được một hợp số $$n$$ luôn có ước dương khác $$1$$ nhỏ hơn hoặc bằng $$\sqrt{n}$$.
Từ đó, ta chỉ cần duyệt $$i$$ từ $$2$$ đến $$\sqrt{n}$$:
- Nếu tồn tại $$i$$ sao cho $$n$$ chia hết cho $$i$$ thì $$n$$ là hợp số.
- Nếu không tồn tại thì $$n$$ là số nguyên tố.
Tất nhiên, nếu $$n < 2$$ thì $$n$$ không phải là số nguyên tố.

```cpp
bool isPrime(long long n) {
    if (n < 2) return false;
    for (long long i = 2; i*i <= n; ++i)
        if (n % i == 0) return false;
    return true;
}
```

**Một thuật toán tốt hơn:**
Để tối ưu thêm thuật toán trên thì đầu tiên ta kiểm tra:
- Nếu $$n < 2$$ thì $$n$$ không phải là số nguyên tố.
- Nếu $$n \geq 2$$ và $$n < 4$$ thì $$n$$ là số nguyên tố.
- Nếu $$n \geq 4$$ và $$n$$ chia hết cho $$2$$ hoặc $$3$$ thì $$n$$ không phải là số nguyên tố.
Nếu $$n$$ không nằm trong các trường hợp trên thì khi đó chỉ cần kiểm tra xem $$n$$ có chia hết cho các số từ $$2$$ đến $$\sqrt{n}$$ có dạng $$6k+1$$ hoặc $$6k+5$$ (hay $$6k-1$$) hay không (không cần kiểm tra các số có dạng $$6k$$, $$6k+2$$, $$6k+3$$, $$6k+4$$ vì $$n$$ không chia hết cho $$2$$ và $$3$$).
Trong $$6$$ số liên tiếp, thay vì kiểm tra $$6$$ số chỉ cần kiểm tra $$2$$ số trong đó nên thời gian xử lí giảm đi $$3$$ lần.

```cpp
bool isPrime(long long n) {
    if (n < 2) return false;
    if (n < 4) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (long long i = 5; i*i <= n; i+=6)
        if (n % i == 0 || n % (i+2) == 0) return false;
    return true;
}
```

## $$O(Klog(N)) / O(Klog(N)^2)$$ - Miller {#miller}
Ta có thể dùng thuật toán Miller để kiểm tra các số nguyên tố 64bit với độ phức tạp nhỏ.

Code ở [**đây**](/so-hoc/miller)