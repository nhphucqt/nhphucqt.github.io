---
title: Tính tích ước số của một số
category: subpage
permalink: /so-hoc/eratosthenes/:title/
keywords: subpage, trang con, giai thich, giải thích, so hoc, số học, number theory, explanation, tich uoc so, tích ước số, phan tich thua so nguyen to, phân tích thừa số nguyên tố, factorize, factorization
---

## Tính tích ước số của $$n$$:

Đặt $$n = {x_1}^{y_1} \cdot {x_2}^{y_2} \cdot {x_3}^{y_3} \ldots {x_k}^{y_k}$$ với $$x_i$$ là thừa số nguyên tố của $$n$$, $$y_i$$ là số mũ của $$x_i$$ $$(1 \le i \le k)$$

Đặt $$c$$ là số lượng ước của $$n$$, ta có:

$$\large c = (y_1+1) \cdot (y_2+1) \cdot (y_3)+1) \ldots (y_k+1)$$

Các ước của $$n$$ có dạng: $${x_1}^{z_1} \cdot {x_2}^{z_2} \cdot {x_3}^{z_3} \ldots {x_k}^{z_k} \hspace{3mm} (0 \le z_k \le y_k)$$

Sử dụng phương pháp tổ hợp, ta có số lượng ước của $$n$$ mà khi phân tích ra thừa số nguyên tố có $${x_i}^{z_i}$$ là:

$$\large {d}_{ {x_i}^{z_i} } = \frac{(y_1+1) \cdot (y_2+1) \cdot (y_3+1) \ldots (y_k+1)}{y_i+1} = \frac{c}{y_i+1}$$

hay 

$$\large {d}_{ {x_i}^0 } = {d}_{ {x_i}^1 } = {d}_{ {x_i}^2 } = \ldots = {d}_{ {x_i}^{y_i} } = \frac{c}{y_i+1} = D_{x_i}$$

### Ví dụ

<p style="display:flex">
$$
\begin{align*}
&n = 12 = 2^2 \cdot 3^1 \\
&d_{2^0} = \frac{(2+1)(1+1)}{(2+1)} = 2 \hspace{3mm} (2^0 \cdot 3^0 = 1; 2^0 \cdot 3^1 = 3) \\
&d_{2^1} = \frac{(2+1)(1+1)}{(2+1)} = 2 \hspace{3mm} (2^1 \cdot 3^0 = 2; 2^1 \cdot 3^1 = 6) \\
&d_{2^2} = \frac{(2+1)(1+1)}{(2+1)} = 2 \hspace{3mm} (2^2 \cdot 3^0 = 4; 2^2 \cdot 3^1 = 12)
\end{align*}
$$
</p>

Khi các ước của $$n$$ nhân lại với nhau, ta gộp các thừa số nguyên tố chung lại với nhau:

Đối với tích các thừa số nguyên tố chung $$t_{x_i}$$, ta có:

<p style="display:flex;">
$$
\large
\begin{align*}
t_{x_i} &= {({x_i}^0)}^{ D_{x_i} } \cdot {({x_i}^1)}^{ D_{x_i} } \cdot {({x_i}^2)}^{ D_{x_i} } \ldots {({x_i}^{y_i})}^{ D_{x_i} } \\
&= ({x_i}^0 \cdot {x_i}^1 \cdot {x_i}^2 \ldots {x_i}^{y_i})^{ D_{x_i} } \\
&= ({x_i}^{0+1+2+ \ldots +y_i})^{ D_{x_i} } \\
&= \left({x_i}^{ \frac{(y_i+1)y_i}{2} }\right)^{ D_{x_i} } \\
&= \left({x_i}^{ \frac{(y_i+1)y_i}{2} }\right)^{ \frac{c}{y_i+1} } \hspace{3mm} (vì \ D_{x_i} = \frac{c}{y_i+1}) \\
&= \left({x_i}^{y_i}\right)^{ \frac{c}{2} }
\end{align*}
$$
</p>

### Ta có tích ước số của $$n$$ là:

$$\large P = \prod_{i=1}^{k} t_{x_i}$$

$$
\large 
\begin{align*}
&= \left({x_1}^{y_1}\right)^{ \frac{c}{2} } \cdot \left({x_2}^{y_2}\right)^{ \frac{c}{2} } \cdot \left({x_3}^{y_3}\right)^{ \frac{c}{2} } \ldots \left({x_k}^{y_k}\right)^{ \frac{c}{2} } \\
&= \left({x_1}^{y_1}{x_2}^{y_2} {x_3}^{y_3} \ldots {x_k}^{y_k}\right)^{ \frac{c}{2} } \\
&= n^{ \frac{c}{2} }
\end{align*}
$$

### Vậy tích ước số của $$n$$ là: $$P = n^{ \frac{c}{2} }$$

<p style="display:flex;">
$$
\large
\Rightarrow P =
\begin{cases}
n^{ \frac{c}{2} }&, n \ne a^2 \\
n^{\lfloor \frac{c}{2} \rfloor} \cdot \sqrt{n}&, n = a^2 \\
\end{cases}
\hspace{3mm} (a \in \mathbb{Z})
$$
</p>

### Ví dụ

* Với $$n = 12 = 2^2 \cdot 3^1$$
<p style="display:flex;">
Ta có:
$$
\begin{align*} 
c &= (2+1)(1+1) = 6 \\ 
P &= 2^{ \frac{6}{2} } = 12^3 = 1728 \ (= 1 \cdot 2 \cdot 3 \cdot 4 \cdot 6 \cdot 12)
\end{align*}
$$
</p>

* Với $$n = 36 = 2^2 \cdot 3^2$$
<p style="display:flex;">
Ta có:
$$
\begin{align*}
c &= (2+1)(2+1) = 9 \\
P &= 36^{ \frac{9}{2} } = 36^4.5 = 10077696 \ (= 1 \cdot 2 \cdot 3 \cdot 4 \cdot 6 \cdot 9 \cdot 12 \cdot 18 \cdot 36) \\
&= 36^{ \lfloor \frac{9}{2} \rfloor } \cdot \sqrt{36} = 10077696
\end{align*}
$$
</p>