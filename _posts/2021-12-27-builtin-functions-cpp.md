---
title: Một số hàm builtin thường dùng trong lập trình thi đấu
category: doc-them
keywords: doc them, đọc thêm, more, builtin function, ham, hàm, bit
---

<div class="table-of-contents" markdown="1">
* [`int __builtin_popcount(unsigned int)`](#popcount)
* [`int __builtin_parity(unsigned int)`](#parity)
* [`int __builtin_ctz(unsigned int)`](#ctz)
* [`int __builtin_clz(unsigned int)`](#clz)
    * [Ứng dụng của `__builtin_clz()`](#ung-dung-clz)
* [Nguồn tham khảo](#tham-khao)
</div>

## `int __builtin_popcount(unsigned int)` {#popcount}

Hàm `__builtin_popcount(x)` trả về số lượng bit $$1$$ trong $$x$$

```cpp
// 00000000000000000000000000100110
cout << __builtin_popcount(38) << '\n'; // 3

// 00000000000000000000100100101001
cout << __builtin_popcount(2345) << '\n'; // 5

// 00000000000000000000000000000000
cout << __builtin_popcount(0) << '\n'; // 0
```

Ngoài ra còn có 2 hàm tương tự:
* `int __builtin_popcountl(unsigned long)`
* `int __builtin_popcountll(unsigned long long)`

## `int __builtin_parity(unsigned int)` {#parity}

Hàm `__builtin_parity(x)` trả về số lượng bit $$1$$ trong $$x$$ $$\bmod 2$$

```cpp
// 00000000000000000000000000100111
cout << __builtin_parity(39) << '\n'; // 0

// 00000000000000000000100100101001
cout << __builtin_parity(2345) << '\n'; // 1

// 00000000000000000000000000000000
cout << __builtin_parity(0) << '\n'; // 0
```

Ngoài ra còn có 2 hàm tương tự:
* `int __builtin_parityl(unsigned long)`
* `int __builtin_parityll(unsigned long long)`

## `int __builtin_ctz(unsigned int)` {#ctz}

Hàm `__builtin_ctz(x)` *(count trailing zeros)* trả về số lượng bit $$0$$ ở cuối, bắt đầu từ bit thấp nhất (bit $$0$$), nếu $$x = 0$$ thì trả về $$32$$

```cpp
// 00000000010000001000010110110000
cout << __builtin_ctz(4228528) << '\n'; // 4

// 00000000000100010010011000110011
cout << __builtin_ctz(1123891) << '\n'; // 0

// 00000000000000000000000000000000
cout << __builtin_ctz(0) << '\n'; // 32
```

Ngoài ra còn có 2 hàm tương tự:
* `int __builtin_ctzl(unsigned long)`: tùy vào bộ dịch mà `long` dùng 32bit hoặc 64bit, nếu $$x = 0$$ thì trả về $$32$$ hoặc $$64$$
* `int __builtin_ctzll(unsigned long long)`: nếu $$x = 0$$ thì trả về $$64$$

## `int __builtin_clz(unsigned int)` {#clz}

Hàm `__builtin_clz(x)` *(count leading zeros)* trả về số lượng bit $$0$$ ở đầu, bắt đầu từ bit cao nhất (bit $$31$$), nếu $$x = 0$$ thì trả về $$32$$

```cpp
// 00000000010000001000010110110000
cout << __builtin_clz(4228528) << '\n'; // 9

// 11000000100100010010011000110011
cout << __builtin_clz(3230737971) << '\n'; // 0

// 00000000000000000000000000000000
cout << __builtin_clz(0) << '\n'; // 32;
```

Ngoài ra còn có 2 hàm tương tự:
* `int __builtin_clzl(unsigned long)`: tùy vào bộ dịch mà `long` dùng 32bit hoặc 64bit, đếm bắt đầu từ bit cao nhất (bit $$31$$ hoặc bit $$63$$), nếu $$x = 0$$ thì trả về $$32$$ hoặc $$64$$
* `int __builtin_clzll(unsigned long long)`: đếm bắt đầu từ bit cao nhất (bit $$63$$), nếu $$x = 0$$ thì trả về $$64$$

### Ứng dụng của `__builtin_clz()` {#ung-dung-clz}

#### Tính $$\lfloor log2(x) \rfloor$$
* Đối với số 32bit: 

```cpp
cout << 31 - __builtin_clz(x) << '\n';
```

* Đối với số 64bit:

```cpp
cout << 63 - __builtin_clzll(x) << '\n';
```

## Nguồn tham khảo {#tham-khao}
* [https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html](https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html)