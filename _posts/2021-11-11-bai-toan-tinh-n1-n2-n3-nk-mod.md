---
title: Bài toán tính (n^1 + n^2 + n^3 + ... + n^k) mod MOD
category: problem
keywords: problem, van de, vấn đề, bai toan, bài toán, so hoc, số học, number theory, modulo, mod
---

## Bài toán:

**Dữ liệu:** Cho 3 số nguyên dương $$n, k \ (n, k \le 10^{18})$$ và $$MOD \ (MOD \le 2 \cdot 10^9)$$.

**Yêu cầu:** Tính $$(1 + n^1 + n^2 + n^3 + \ldots + n^k) \bmod MOD$$.

**input:** gồm một dòng duy nhất là 3 số $$n$$, $$k$$ và $$MOD$$.

**output:** gồm một dòng duy nhất là kết quả của bài toán.

**Ví dụ:**

<div class="input-output-block" markdown="1">
**input**
```
2 3 123
```
**output:**
```
15
```
</div>
<br style="content:''">
<div class="input-output-block" markdown="1">
**input**
```
123456789123456789 3 1000000007
```
**output**
```
701012881
```
</div>

**Subtasks:**
* subtask 1: $$k \le 10^6$$
* subtask 2: $$MOD$$ là số nguyên tố
* subtask 3: không giới hạn gì thêm

[**Solution**](./solution/)