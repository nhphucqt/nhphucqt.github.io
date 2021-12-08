---
title: Phân tích N ra thừa số nguyên tố
category: problem
keywords: problem, van de, vấn đề, bai toan, bài toán, so hoc, số học, number theory, loi giai, lời giải, solution, phan tich thua so nguyen to, phân tích thừa số nguyên tố, factorize, factorization
---

**Bài toán:** Cho một số nguyên dương $$n$$, hãy phân tích $$n$$ ra thừa số nguyên tố

**Input:** Gồm một dòng duy nhất là số nguyên dương $$n \ (1 \le n \le 10^{12})$$.

**Output:** Gồm $$m$$ dòng, mỗi dòng in ra 2 số $$x, y$$ cách nhau một dấu cách được sắp xếp theo thứ tự tăng dần của $$x$$ với $$m$$ là số lượng thừa số nguyên tố của $$n, x$$ là thừa số nguyên tố của $$n$$ và $$y$$ là số mũ của $$x$$.

**Ví dụ:**

<div class="input-output-block" markdown="1">
**input**
```
2250
```
**output**
```
2 1
3 2
5 3
```
</div>

<br style="content:''">

<details class="spoiler" markdown="1">
<summary><strong>Ý tưởng</strong></summary>
* Để phân tích $$n$$ ra thừa số nguyên tố ta chia $$n$$ cho ước nguyên tố nhỏ nhất của $$n$$ và cứ lặp lại bước đó cho đến khi $$n = 1$$.
* Ví dụ:

<img src="/assets/images/phan-tich-thua-so-nguyen-to.svg" width="60%" class="center">
<br>

<details class="spoiler" markdown="1">
<summary>code</summary>
```cpp
{% include cpp/factorization.cpp %}
```
</details>
</details>