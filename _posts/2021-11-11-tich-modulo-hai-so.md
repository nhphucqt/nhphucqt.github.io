---
title: Tích modulo hai số
category: so-hoc
---

<div class="table-of-contents" markdown="1">
* [Sử dụng phép nhân Ấn Độ](#phep-nhan-an-do)
* [Sử dụng kiểu dữ liệu trong C++](#kieu-du-lieu-cpp)
* [Nguồn tham khảo](#nguon-tham-khao)
</div>

Dưới đây là một số cách tính $$a \cdot b \bmod m \ (a, b \geq 0, m > 0)$$ mà mình sưu tầm được:

## Sử dụng phép nhân Ấn Độ {#phep-nhan-an-do}

**Thuật toán:** $$a \cdot b \bmod m = a \cdot (\lfloor b/2 \rfloor + \lfloor b/2 \rfloor + m \bmod 2) \bmod m$$

Đây là code dựa trên [**câu trả lời của Dana Jacobsen**](https://www.quora.com/How-can-I-execute-A-*-B-mod-C-without-overflow-if-A-and-B-are-lesser-than-C/answer/Dana-Jacobsen) trên Quora:

```cpp
static uint64_t mul(uint64_t a, uint64_t b, uint64_t m) {
    uint64_t r = 0;
    if (a >= m) a %= m;
    if (b >= m) b %= m;
    if ((a|b) < (1ull << 32)) return (a*b) % m;
    if (a < b) { uint64_t t = a; a = b; b = t; }
    if (m <= (1ull << 63)) {
        while (b > 0) {
            if (b & 1) { r += a;  if (r >= m) r -= m; }
            b >>= 1;
            if (b) { a += a;  if (a >= m) a -= m; }
        }
    } else {
        while (b > 0) {
            if (b & 1) r = ((m-r) > a) ? r+a : r+a-m;    /* r = (r + a) % n */
            b >>= 1;
            if (b) a = ((m-a) > a) ? a+a : a+a-m;    /* a = (a + a) % n */
        }
    }
    return r;
}
```

Ở code trên có sử dụng phép cộng tràn số ở 2 câu lệnh:
```cpp
if (b & 1) r = ((m-r) > a) ? r+a : r+a-m;    /* r = (r + a) % n */
```
```cpp
if (b) a = ((m-a) > a) ? a+a : a+a-m;    /* a = (a + a) % n */
```

Mặc dù là có bị tràn số nhưng nó không ảnh hưởng đến kết quả, có thể đọc ở [**đây**](http://www.cplusplus.com/articles/DE18T05o/) để có thể hiểu tại sao.

Code ở trên là code tối ưu có thể tính phép nhân modulo của các số 64bit $$(< 2^{64})$$. Nhược điểm là code khá dài, vậy nên có thể rút gọn lại như sau:

```cpp
uint64_t mul(uint64_t a, uint64_t b, uint64_t m) {
    a %= m;
    b %= m;
    if ((a|b)<(1ull<<32)) return a*b%m;
    uint64_t res = 0;
    while (b>0) {
        if (b&1) res = (m-res>a) ? res+a : res+a-m;
        a = (m-a>a) ? a<<1 : (a<<1)-m;
        b >>= 1;
    }
    return res;
}
```

Code này vẫn có thể tính các số 64bit nhưng gọn hơn, trong code vẫn sử dụng phép cộng tràn số, nếu không muốn có tràn số thì có thể chỉnh lại như sau:

```cpp
uint64_t mul(uint64_t a, uint64_t b, uint64_t m) {
    a %= m;
    b %= m;
    if ((a|b)<(1ull<<32)) return a*b%m;
    uint64_t res = 0;
    while (b>0) {
        if (b&1) res = (m-res>a) ? res+a : res-(m-a);
        a = (m-a>a) ? a<<1 : a-(m-a);
        b >>= 1;
    }
    return res;
}
```

Mặc dù không có tràn số nhưng nhìn chung thì code này không tối ưu bằng code có cộng tràn số.

Đây là code cơ bản sử dụng thuật toán này:

```cpp
uint64_t mul(uint64_t a, uint64_t b, uint64_t m) {
    a %= m; b %= m;
    uint64_t res = 0;
    for (; b>0; a=(a<<1)%m,b>>=1) {
        if (b&1) res = (res+a)%m;
    }
    return res;
}
```

Ưu điểm của code này là đơn giản hơn và code nhanh, nhưng nhược điểm là chỉ tính với số $$m \le 2^{63}$$ và thời gian xử lí lại chậm hơn nhiều so với các code trên.

Ngoài ra còn cách tính khá thú vị dựa trên [**câu trả lời của Jonas Oberhauser**](https://www.quora.com/How-can-I-execute-A-*-B-mod-C-without-overflow-if-A-and-B-are-lesser-than-C/answer/Jonas-Oberhauser):

```cpp
static uint64_t slowModulo(uint64_t a, uint64_t b, uint64_t m) {
    uint64_t r = 0;
    uint64_t C_down = m>>1;
    uint64_t C_up = m - C_down;
    while (b > 0) {
        if (b&1) r = ((r >= m - a) ? (a >= C_up ? a - C_up + r : r - C_up + a) - C_down : r+a);
        if (a >= C_up) a = (a-C_down)+(a-C_up);
        else a = a+a;
        b >>= 1;
    }
    return r;
}

static uint64_t fastModulo(uint64_t a, uint64_t b, uint64_t m, uint64_t stepSize) {
    uint64_t mask = (1 << stepSize) - 1;
    uint64_t r = 0;
    while (b > 0) {
        r += a * (b&mask); r %= m;
        a <<= stepSize; a %= m;
        b >>= stepSize;
    }
    return r;
}

static uint64_t mul(uint64_t a, uint64_t b, uint64_t m) {
    if (a>=m) a %= m;
    if (b>=m) b %= m;
    if ((a|b) < (1ULL << 32)) return (a*b) % m;
    if (a < b) { uint64_t t = a; a = b; b = t; }
    int stepSize = __builtin_clz((uint32_t)(m>>32));
    if (stepSize == 0) return slowModulo(a,b,m);
    return fastModulo(a,b,m,stepSize);
}
```

Hàm `slowModulo` chính là thuật toán phép nhân Ấn Độ, code tránh tràn số nhưng code khá dài và chậm hơn code không tràn số ở trên.

Ở hàm `fastModulo`, thuật toán cơ bản cũng cùng ý tưởng với phép nhân Ấn Độ nhưng số lượng bit đầu của $$b$$ nhân vào $$a$$ từng vòng lặp lớn hoặc bằng $$1$$, trong khi phép nhân Ấn Độ chỉ lấy $$1$$ bit đầu của $$b$$ nhân với $$a$$, vì vậy nếu số lượng bit càng lớn (số càng nhỏ) thì hàm sẽ chạy càng nhanh, nhưng nếu hàm `fastModulo` chỉ lấy khoảng $$1$$, $$2$$ bit thì có thể sẽ không tối ưu hơn code của [**Dana Jacobsen**](https://www.quora.com/How-can-I-execute-A-*-B-mod-C-without-overflow-if-A-and-B-are-lesser-than-C/answer/Dana-Jacobsen) , nhưng khi tính các số nằm trong khoảng từ 32bit đến khoảng 55bit ~ 60bit thì sẽ tối ưu hơn.

## Sử dụng kiểu dữ liệu trong C++ {#kieu-du-lieu-cpp}

C++ có hỗ trợ một số kiểu dữ liệu có thể lưu được giá trị các số lớn, khi sử dụng thì tốc độ tính toán sẽ nhanh hơn rất nhiều. Sau đây mình tìm được 2 cách để tính $$a \cdot b \bmod c$$ bằng cách sử dụng kiểu dữ liệu có sẵn trong C++:

### __uint128_t

```cpp
uint64_t mul(uint64_t a, uint64_t b, uint64_t m) {
    auto res = (__uint128_t) a * b % m;
    return (uint64_t) res;
}
```

Trong C++ có hỗ trợ kiểu dữ liệu **\_\_int128\_t** cho phép lưu các số nguyên 128bit, nếu muốn tính các số nhỏ hơn $$2^{64}$$ thì phải dùng **\_\_uint128\_t**.

* **Ưu điểm:** code đơn giản, xử lí nhanh.
* **Nhược điểm:** có một số bộ dịch không hỗ trợ kiểu dữ liệu này.

### long double

```cpp
uint64_t mul(uint64_t a, uint64_t b, uint64_t m) {
    if (a >= m) a %= m;
    if (b >= m) b %= m;
    uint64_t q = (long double) a * b / m;
    uint64_t r = a * b - q * m;
    return r;
}
```

**Thuật toán:** với mọi số nguyên không âm $$a, b, m$$, ta luôn có: 

$$a \cdot b = (\lfloor a \cdot b / m \rfloor) \cdot m + (a \cdot b \bmod m) \Rightarrow a \cdot b \bmod m = a \cdot b - (\lfloor a \cdot b / m \rfloor) \cdot m$$

Trong C++ có hỗ trợ kiểu dữ liệu **long double** cho phép lưu các số thực lớn, do đó có thể tính được $$q = a \cdot b / m$$.

Trong code có [**phép nhân tràn số**](http://www.cplusplus.com/articles/DE18T05o/), kết quả sẽ không bị ảnh hưởng bởi vì nếu $$a \cdot b$$ bị tràn số thì giá trị $$a \cdot b$$ sẽ chỉ lấy 64 bit đầu, tương đương với $$(a \cdot b) \& (2^{64}-1) = a \cdot b \bmod 2^{64}$$ (vì giá trị không âm), $$q \cdot m$$ cũng tương tự, có nghĩa rằng $$a \cdot b - q \cdot m$$ sẽ bằng $$(a \cdot b - q \cdot m) \bmod 2^{64}$$ , nhưng vì $$a \cdot b - q \cdot m = a \cdot b \bmod m < m < 2^{64}$$ nên việc $$\bmod 2^{64}$$ không ảnh hưởng gì đến kết quả cả.

* **Ưu điểm:** code đơn giản, có tốc độ xử lí nhanh nhất trong các code ở trên.

* **Nhược điểm:** vì có dùng số thực nên có thể sẽ bị sai số dẫn đến kết quả sai, cách này chỉ đúng khi và chỉ khi q được tính đúng, số càng lớn thì sai số càng lớn nên khả năng tính sai càng lớn, sau đây là kết quả thống kê sau khi chạy $$1000$$ lần, mỗi lần chạy $$10^6$$ test random đối với các số:

* $$< 2^{64}$$: trung bình khoảng $$74934$$ test sai / $$10^6$$ test , chiếm khoảng $$7.5\%$$
* $$\le 2^{63}$$: trung bình khoảng $$37501$$ test sai / $$10^6$$ test, chiếm khoảng $$3.75\%$$
* $$\le 2^{60}$$: trung bình khoảng $$4687$$ test sai / $$10^6$$ test, chiếm khoảng $$0.47\%$$
* $$\le 10^{18}$$: trung bình khoảng $$3977$$ test sai / $$10^6$$ test, chiếm khoảng $$0.4\%$$
* $$\le 10^{16}$$: trung bình khoảng $$41$$ test sai / $$10^6$$ test, chiếm khoảng $$0.004\%$$
* $$\le 10^{14}$$: trung bình khoảng $$0.4$$ test sai / $$10^6$$ test, chiếm khoảng $$0.00004\%$$

Từ thống kê trên, tùy thuộc vào dữ liệu cần xử lí mà ta sẽ lựa chọn được cách tính thích hợp.

Về thời gian chạy của các code, có thể tham khảo thống kê tại [**đây**](https://docs.google.com/spreadsheets/d/143Fy0PL0pZIACbzgtvubTscIAXbCD89zvsrCTYkjRK4/edit?usp=sharing), trước khi xem thì nên xem đọc **code** ở dưới trước để biết tên mỗi hàm.

<details class="spoiler" markdown="1">
<summary>code</summary>
```cpp
// m < 2^64, có cộng tràn số, tối ưu, code dài, phức tạp
static uint64_t mul(uint64_t a, uint64_t b, uint64_t m) {
    uint64_t r = 0;
    if (a >= m) a %= m;
    if (b >= m) b %= m;
    if ((a|b) < (1ull << 32)) return (a*b) % m;
    if (a < b) { uint64_t t = a; a = b; b = t; }
    if (m <= (1ull << 63)) {
        while (b > 0) {
            if (b & 1) { r += a;  if (r >= m) r -= m; }
            b >>= 1;
            if (b) { a += a;  if (a >= m) a -= m; }
        }
    } else {
        while (b > 0) {
            if (b & 1) r = ((m-r) > a) ? r+a : r+a-m;    /* r = (r + a) % n */
            b >>= 1;
            if (b) a = ((m-a) > a) ? a+a : a+a-m;    /* a = (a + a) % n */
        }
    }
    return r;
}

// m < 2^64, có cộng tràn số, đơn giản hơn mul
uint64_t mul1(uint64_t a, uint64_t b, uint64_t m) {
    a %= m;
    b %= m;
    if ((a|b)<(1ull<<32)) return a*b%m;
    uint64_t res = 0;
    while (b>0) {
        if (b&1) res = (m-res>a) ? res+a : res+a-m;
        a = (m-a>a) ? a<<1 : (a<<1)-m;
        b >>= 1;
    }
    return res;
}

// m < 2^64, không có cộng tràn số, đơn giản hơn mul
uint64_t mul2(uint64_t a, uint64_t b, uint64_t m) {
    a %= m;
    b %= m;
    if ((a|b)<(1ull<<32)) return a*b%m;
    uint64_t res = 0;
    while (b>0) {
        if (b&1) res = (m-res>a) ? res+a : res-(m-a);
        a = (m-a>a) ? a<<1 : a-(m-a);
        b >>= 1;
    }
    return res;
}

// m <= 2^63, đơn giản, chậm
uint64_t mul3(uint64_t a, uint64_t b, uint64_t m) {
    a %= m; b %= m;
    uint64_t res = 0;
    for (; b>0; a=(a<<1)%m,b>>=1) {
        if (b&1) res = (res+a)%m;
    }
    return res;
}

static uint64_t slowModulo(uint64_t a, uint64_t b, uint64_t m) {
    uint64_t r = 0;
    uint64_t C_down = m>>1;
    uint64_t C_up = m - C_down;
    while (b > 0) {
        if (b&1) r = ((r >= m - a) ? (a >= C_up ? a - C_up + r : r - C_up + a) - C_down : r+a);
        if (a >= C_up) a = (a-C_down)+(a-C_up);
        else a = a+a;
        b >>= 1;
    }
    return r;
}

// stepSize càng lớn thì chạy càng nhanh
static uint64_t fastModulo(uint64_t a, uint64_t b, uint64_t m, uint64_t stepSize) {
    uint64_t mask = (1 << stepSize) - 1;
    uint64_t r = 0;
    while (b > 0) {
        r += a * (b&mask); r %= m;
        a <<= stepSize; a %= m;
        b >>= stepSize;
    }
    return r;
}

// m < 2^64, code dài, phức tạp
static uint64_t mul4(uint64_t a, uint64_t b, uint64_t m) {
    if (a>=m) a %= m;
    if (b>=m) b %= m;
    if ((a|b) < (1ULL << 32)) return (a*b) % m;
    if (a < b) { uint64_t t = a; a = b; b = t; }
    int stepSize = __builtin_clz((uint32_t)(m>>32));
    if (stepSize == 0) return slowModulo(a,b,m);
    return fastModulo(a,b,m,stepSize);
}

// m < 2^64, nhanh, một số bộ dịch không hỗ trợ
uint64_t mul5(uint64_t a, uint64_t b, uint64_t m) {
    auto res = (__uint128_t) a * b % m;
    return (uint64_t) res;
}

// m < 2^64, nhanh hơn mul5, nhưng có thể bị sai số vì dùng số thực
uint64_t mul6(uint64_t a, uint64_t b, uint64_t m) {
    if (a >= m) a %= m;
    if (b >= m) b %= m;
    uint64_t q = (long double) a * b / m;
    uint64_t r = a * b - q * m;
    return r;
}
```
</details>

## Nguồn tham khảo {#nguon-tham-khao}

* [Câu hỏi trên Quora](https://www.quora.com/How-can-I-execute-A-*-B-mod-C-without-overflow-if-A-and-B-are-lesser-than-C)
* [Integer overflow - cplusplus](http://www.cplusplus.com/articles/DE18T05o/)
* [https://en.wikipedia.org/wiki/Two's_complement](https://en.wikipedia.org/wiki/Two's_complement)
* [https://en.wikipedia.org/wiki/Integer_overflow](https://en.wikipedia.org/wiki/Integer_overflow)