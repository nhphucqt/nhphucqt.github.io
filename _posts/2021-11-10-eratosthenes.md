---
title: Eratosthenes
category: so-hoc
---

<div class="table-of-contents" markdown="1">
* [Sàng nguyên tố](#sang-nguyen-to)
* [Tìm ước nguyên tố nhỏ nhất của một số](#uoc-nguyen-to-nho-nhat)
* [Tìm ước nguyên tố lớn nhất của một số](#uoc-nguyen-to-lon-nhat)
* [Phân tích một số ra thừa số nguyên tố](#phan-tich-thua-so-nguyen-to)
* [Tính số lượng ước của một số](#so-luong-uoc)
* [Tính tổng ước số của một số](#tong-uoc-so)
* [Tính tích ước số của một số](#tich-uoc-so)
* [Sàng nguyên tố trên đoạn [L..R]](#sang-tren-doan)
* [Mở rộng - Linear Sieve](#mo-rong)
</div>

## Sàng nguyên tố {#sang-nguyen-to}
Sàng Eratosthenes: [https://vi.wikipedia.org/wiki/Sàng_Eratosthenes](https://vi.wikipedia.org/wiki/Sàng_Eratosthenes)

**Sàng Eratosthenes là thuật toán cơ bản và làm cơ sở cho các thuật toán ở dưới.**

Độ phức tạp: $$O(Nlog(N))$$

```cpp
const int N = 1e7+7;
bool isPrime[N];

void sieve() {
    memset(isPrime, true, sizeof isPrime);
    isPrime[0] = isPrime[1] = false;
    for (int i = 2; i*i < N; ++i) 
        if (isPrime[i])
            for (int j = i*i; j < N; j+=i)  
                isPrime[j] = false;
}
```

## Tìm ước nguyên tố nhỏ nhất của một số {#uoc-nguyen-to-nho-nhat}
Độ phức tạp: $$O(Nlog(N))$$

```cpp
const int N = 1e7+7;
int lp[N];

void sieve() {
    for (int i = 1; i < N; ++i) lp[i] = i;
    for (int i = 2; i*i < N; ++i) 
        if (lp[i] == i)
            for (int j = i*i; j < N; j+=i)  
                if (lp[j] == j) lp[j] = i;
}
```

## Tìm ước nguyên tố lớn nhất của một số {#uoc-nguyen-to-lon-nhat}
Độ phức tạp: $$O(Nlog(N))$$

```cpp
const int N = 1e7+7;
int mp[N];

void sieve() {
    for (int i = 1; i < N; ++i) mp[i] = i;
    for (int i = 2; i*i < N; ++i) 
        if (mp[i] == i)
            for (int j = i*i; j < N; j+=i)
                if (i < mp[j]) mp[j] = mp[j/i];
                else mp[j] = i;
}
```

## Phân tích một số ra thừa số nguyên tố {#phan-tich-thua-so-nguyen-to}

```cpp
const int N = 1e7+7;
int lp[N];

void sieve() {
    for (int i = 2; i < N; ++i) lp[i] = i;
    for (int i = 2; i*i < N; ++i) 
        if (lp[i] == i)
            for (int j = i*i; j < N; j+=i)
                if (lp[j] == j) lp[j] = i;
}

vector<pair<int,int>> factorize(int n) {
    vector<pair<int,int>> fac;
    while (n > 1) {
        int p = lp[n];
        fac.push_back(make_pair(p,0));
        while (n % p == 0) {
            fac.back().second++;
            n /= p;
        }
    }
    return fac;
}
```

* **Độ phức tạp:**
    * Tiền xử lí: $$O(Nlog(N))$$
    * Phân tích thừa số nguyên tố: $$O(log(N))$$
* Để phân tích một số $$n$$ rất lớn ra thừa số nguyên tố, ta phải dùng thuật toán khác, thuật toán này sẽ được giải thích ở [**đây**](/problem/phan-tich-thua-so-nguyen-to/)

## Tính số lượng ước của một số {#so-luong-uoc}
Độ phức tạp: $$O(Nlog(N))$$

```cpp
const int N = 1e6+7;
int numDiv[N];

void sieve() {
    for (int i = 1; i < N; ++i)
    for (int j = i; j < N; j+=i)
        numDiv[j]++;
}
```

* **Cải tiến thuật toán:**

Để tính số lượng ước của một số $$n$$, ta làm như sau:
* Phân tích $$n$$ ra thừa số nguyên tố: $$n = {x_1}^{y_1} \times {x_2}^{y_2} \times {x_3}^{y_3} \times \ldots \times {x_k}^{y_k}$$, trong đó $$x_i$$ là thừa số nguyên tố của $$n$$, $$y_i$$ là số mũ của $$x_i$$ $$(1 \le i \le k)$$
* Sử dụng công thức tính số lượng ước dựa vào số mũ của các thừa số nguyên tố:

$$numDiv(n) = (y_1+1) \times (y_2+1) \times (y_3+1) \times \ldots \times (y_k+1)$$

```cpp
const int N = 1e7+7;
int lp[N];

void sieve() {
    for (int i = 2; i < N; ++i) lp[i] = i;
    for (int i = 2; i*i < N; ++i)
        if (lp[i] == i)
            for (int j = i*i; j < N; j+=i)
                if (lp[j] == j) lp[j] = i;
}

int numDiv(int n) {
    int num = 1;
    while (n > 1) {
        int p = lp[n], cnt = 0;
        while (n % p == 0) {
            cnt++; n /= p;
        }
        num *= cnt+1;
    }
    return num;
}
```

* **Độ phức tạp:**
    * Tiền xử lí: $$O(Nlog(N))$$
    * Tính số lượng ước: $$O(log(N))$$

## Tính tổng ước số của một số {#tong-uoc-so}
Độ phức tạp: $$O(Nlog(N))$$

```cpp
const int N = 1e6+7;
int sumDiv[N];

void sieve() {
    for (int i = 1; i < N; ++i)
    for (int j = i; j < N; j+=i) {
        sumDiv[j] += i;
    }
}
```

**Cải tiến thuật toán:**

Để tính tổng ước số của $$n$$, ta làm như sau:
* Phân tích $$n$$ ra thừa số nguyên tố, đặt $$n = {x_1}^{y_1} \times {x_2}^{y_2} \times {x_3}^{y_3} \times \ldots \times {x_k}^{y_k}$$ với $$x_i$$ là thừa số nguyên tố của $$n$$, $$y_i$$ là số mũ của $$x_i$$ $$(1 \le i \le k)$$
* Sử dụng công thức tính tổng ước số của $$n$$:

$$sumDiv(n) = ({x_1}^0 + {x_1}^1 + {x_1}^2 + \ldots + {x_1}^{y_1}) \times ({x_2}^0 + {x_2}^1 + {x_2}^3 + \ldots + {x_2}^{y_2}) \times \ldots \times ({x_k}^0 + {x_k}^1 + {x_k}^2 + \ldots + {x_k}^{y_k})$$

Ta có thể tính công thức $$x^0 + x^1 + x^2 + \ldots + x^n$$ trong $$O(log(N))$$, xem cách giải tại [**đây**](/problem/bai-toan-tinh-n1-n2-n3-nk-mod/solution/)

```cpp
const int N = 1e7+7;
int lp[N];

void sieve() {
    for (int i = 2; i < N; ++i) lp[i] = i;
    for (int i = 2; i*i < N; ++i)
        if (lp[i] == i)
            for (int j = i*i; j < N; j+=i)
                if (lp[j] == j) lp[j] = i;
}

long long Pow(long long a, int n) {
    long long res = 1;
    for (;n;n>>=1,a*=a)
        if (n&1) res*=a;
    return res;
}

int sumDiv(int n) {
    int sum = 1;
    while (n > 1) {
        int p = lp[n], cnt = 0;
        while (n % p == 0) {
            cnt++; n /= p;
        }
        sum *= (Pow(p, cnt+1)-1) / (p-1);
    }
    return sum;
}
```
* **Độ phức tạp:**
    * Tiền xử lý: $$O(Nlog(N))$$
    * Tính tổng các ước: $$O(log(N)*log(log(N)))$$, vì $$N \approx 10^7$$ nên có thể có thể coi độ phức tạp là $$O(log(N))$$

## Tính tích ước số của một số có modulo {#tich-uoc-so}
Độ phức tạp: $$O(Nlog(N))$$

```cpp
const int MOD = 1e9+7;
const int N = 1e6+7;
int prod[N];

void sieve() {
    for (int i = 0; i < N; ++i) prod[i] = 1;
    for (int i = 1; i < N; ++i)
    for (int j = i; j < N; j+=i) {
        prod[j] = 1LL * prod[j] * i % MOD;
    }
}
```

**Cải tiến thuật toán:**

Ngoài thuật toán trên, ta có thể dùng cách phân tích thành thừa số nguyên tố để đếm số lượng ước của một số từ đó tính được tích ước số của số đó:
* Gọi $$x$$ là số lượng ước của $$n$$
* Tích ước số của $$n$$ là: $$n^{x/2}$$

$$prodDiv(n) = n^{x/2} \bmod MOD$$ nếu $$n$$ không phải là số chính phương

$$prodDiv(n) = n^{\lfloor x/2 \rfloor} \times \sqrt{n} \bmod MOD$$ nếu $$n$$ là số chính phương

Giải thích chi tiết tại [**đây**](./tinh-tich-uoc-so)

```cpp
const int MOD = 1e9+7;
const int N = 1e7+7;
int lp[N];

void sieve() {
    for (int i = 2; i < N; ++i) lp[i] = i;
    for (int i = 2; i*i < N; ++i)
        if (lp[i] == i)
            for (int j = i*i; j < N; j+=i)
                if (lp[j] == j) lp[j] = i;
}

int numDiv(int n) {
    int num = 1;
    while (n > 1) {
        int p = lp[n], cnt = 0;
        while (n % p == 0) {
            cnt++; n /= p;
        }
        num *= cnt+1;
    }
    return num;
}

int Pow(int a, int n) {
    int res = 1;
    for (;n;n>>=1,a=1LL*a*a%MOD)
        if (n&1) res = 1LL*res*a%MOD;
    return res;
}

int mySqrt(int n) {
    int i = sqrt(n);
    while (i*i < n) i++;
    while (i*i > n) i--;
    return i;
}

int prodDiv(int n) {
    int x = numDiv(n);
    return x&1 ? 1LL*Pow(n,x/2)*mySqrt(n)%MOD : Pow(n,x/2);
}
```

* **Độ phức tạp:**
    * Tiền xử lý: $$O(Nlog(N))$$
    * Tính tích ước số: $$O(logN))$$

## Sàng nguyên tố trên đoạn [L..R] {#sang-tren-doan}
Độ phức tạp: $$O(\sqrt{R}*log(R-L+1))$$

```cpp
{% include cpp/range-sieve.cpp %}
```

## Mở rộng - Linear Sieve {#mo-rong}
Ngoài cách sàng $$O(Nlog(N))$$ còn có cách sàng $$O(N)$$, có thể đọc tại [**đây**](https://codeforces.com/blog/entry/54090)