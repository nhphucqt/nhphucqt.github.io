---
title: Sol tính (1 + n^1 + n^2 + n^3 + ... + n^k) mod MOD
category: subpage
permalink: /problem/bai-toan-tinh-n1-n2-n3-nk-mod/solution/
---

**Đề bài:** [Bài toán tính (1 + n^1 + n^2 + n^3 + ... + n^k) mod MOD](./..)

## Subtask 1: 

Chạy $$i$$ từ $$0$$ đến $$k$$, độ phức tạp $$O(K)$$

```cpp
int sub1() {
    int ans = 0, p = 1;
    n %= MOD;
    for (int i = 0; i <= k; ++i) {
        ans = (ans + p) % MOD;
        p = 1LL * p * n % MOD;
    }
    return ans;
}
```

## Subtask 2:

Nếu $$n > 1$$, $$MOD$$ là số nguyên tố và $$(n, MOD) = 1$$, Ta có:

$$1 + n^1 + n^2 + \ldots + n^k = \frac{n^{k+1}-1}{n-1}$$

Vậy nên:

$$
\begin{align*}
&\hspace{7mm}(1 + n^1 + n^2 + \ldots + n^k) \bmod MOD \\
&= \frac{n^{k+1}-1}{n-1} \bmod MOD \\
&=\left(\left(n^{k+1} \bmod MOD - 1\right) \bmod MOD \cdot \left(\frac{1}{n-1} \bmod MOD\right)\right) \bmod MOD
\end{align*}
$$

Sử dụng nghịch đảo modulo bằng [**định lý Fermat nhỏ**](https://vi.wikipedia.org/wiki/Định_lý_nhỏ_Fermat):

$$\frac{1}{n-1} \bmod MOD = (n-1)^{MOD-2} \bmod MOD$$

```cpp
int inv(long long n, int MOD) { // (1/n) % MOD
    return Pow(n%MOD, MOD-2, MOD);
}
```

Từ đó ta có công thức là:

$$\left(\left(n^{k+1} \bmod MOD - 1\right) \bmod MOD \cdot \left((n-1)^{MOD-2} \bmod MOD\right)\right) \bmod MOD$$

**Chú ý: Muốn sử dụng nghịch đảo modulo bằng định lý fermat nhỏ thì phải thỏa mãn điều kiện $$MOD$$ là số nguyên tố và $$(n-1)$$ nguyên tố cùng nhau với $$MOD$$.**

Nếu $$n = 1$$ thì không thể dùng công thức trên mà phải dùng công thức: $$(k+1) \bmod MOD$$

Nếu $$MOD$$ là số nguyên tố mà $$(n-1)$$ không nguyên tố cùng nhau với $$MOD$$ thì ta có thể xét riêng trường hợp này:

* Ta có $$(n-1, MOD) \ne 1 \Rightarrow (n-1) \mathrel{\vdots} MOD \Rightarrow n \bmod MOD = 1$$
* Do đó ta cần xét riêng trường hợp $$n \bmod MOD = 1$$
* Nếu $$n \bmod MOD = 1$$ thì $$n^m \bmod MOD = 1 \mathrel{\forall} m \in \mathbb{N}$$
* Do đó $$1 + n^1 + n^2 + n^3 + \ldots + n^k \equiv k+1 \pmod{MOD}$$

Gộp 2 trường hợp $$n = 1$$ và $$n \bmod MOD = 1$$ lại ta có nếu $$n \bmod MOD = 1$$ thì đáp án chính là $$(k+1) \bmod MOD$$

```cpp
int sub2() {
    if (n % MOD == 1) return (k+1) % MOD;
    int ans = 1LL * (Pow(n%MOD, k+1, MOD) - 1) * inv(n-1, MOD) % MOD;
    if (ans < 0) ans += MOD;
    return ans;
}
```
**Độ phức tạp:** $$O(log(max(K, MOD)))$$

Tính Pow(a, n, MOD) trong $$O(log(N))$$ tại [**đây**](/so-hoc/tinh-luy-thua/)

## Subtask 3:

Ta không sử dụng nghịch đảo modulo mà dùng phương pháp **chia để trị**:

* Đặt $$Cal(n, k) = 1 + n^1 + n^2 + \ldots + n^k$$
* Nếu $$k$$ lẻ:

$$
\begin{align*}
Cal(n, k) &= (1+n) + n^2(1+n) + \ldots + n^{k-1}(1+n) \\
&= (1+n)(1 + n^2 + n^4 + \ldots + n^{k-1}) \\
&= (1+n)(1+ (n^2)^1 + (n^2)^2 + \ldots + (n^2)^{ \frac{k-1}{2} }) \\
&= (1+n)\ Cal\left(n^2, \frac{k-1}{2}\right)
\end{align*}
$$

* Nếu $$k$$ chẵn:

$$
\begin{align*}
Cal(n, k) &= 1 + n(1+n) + n^3(1+n) + \ldots + n^{k-1}(1+n) \\
&= 1 + n(1+n)(1 + n^2 + n^4 + \ldots + n^{k-2}) \\
&= 1 + n(1+n)(1 + (n^2)^1 + (n^2)^2 + \ldots + (n^2)^{ \frac{k-2}{2} }) \\
&= 1 + n(1+n)\ Cal\left(n^2, \frac{k-2}{2}\right)
\end{align*}
$$

Từ đó, ta có:

$$
\begin{cases}
Cal(n, k) = 1 &, k = 0 \\
Cal(n, k) = (n+1)\ Cal\left(n^2, \frac{k-1}{2}\right) &, k \bmod 2 \ne 0 \\
Cal(n, k) = n(n+1)\ Cal\left(n^2, \frac{k-2}{2}\right) &, k \bmod 2 = 0
\end{cases}
$$

**Độ phức tạp:** $$O(log(K))$$

```cpp
int Cal(long long n, long long k) {
    if (k == 0) return 1;
    if (k % 2 != 0) return (1+n)*Cal(n*n%MOD, (k-1)/2) % MOD;
    return (1 + n*(1+n)%MOD * Cal(n*n%MOD, (k-2)/2)) % MOD;
}

int sub3() {
    return Cal(n % MOD, k);
}
```