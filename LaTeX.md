**自动调整括号大小**
> 通过`\left\right`自动调整花括号大小
> $\left\lfloor{\frac{a+c}{b+d}}\right\rfloor$


**空格**

| 命令         | 宽度               | 示例（语法 → 效果）                     |
|--------------|--------------------|----------------------------------------|
| `\,`         | 小空格（约 `1/6 em`） | `a\,b` → $a\,b$                     |
| `\:` 或 `\>` | 中等空格（约 `2/7 em`） | `a\:b` → $a\:b$ 或 $a\>b$          |
| `\;`         | 大空格（约 `5/18 em`） | `a\;b` → $a\;b$                     |
| `\quad`      | 标准空格（`1 em`） | `a\quad b` → $a\quad b$             |
| `\qquad`     | 双倍空格（`2 em`） | `a\qquad b` → $a\qquad b$           |


| 常用字符| 公式 | 实例 |
| ------ | ---- | ---------- |
| 乘号 | `\times` | `a \times b` -> $a \times b$ |
| 点乘 | `\cdot` | `a \cdot b` -> $a \cdot b$ |
| 除号 | `\div` | `a \div b` -> $a \div b$ |
| 分数 | `\frac{}{}` | `\frac{a}{b}` -> $\frac{a}{b}$ |
| 小于 | `<`或`\lt` | `a \lt b` -> $a \lt b$ |
| 大于 | `>`或`\gt` | `a \gt b` -> $a \gt b$ |
| 小于等于 | `\leq`或`\le` | `a \leq b` -> $a \leq b$ |
| 大于等于 | `\geq`或`\ge` | `a \geq b` -> $a \geq b$ |
| 不等于 | `\neq`或`\ne` | `a \neq b` -> $a \neq b$ |
| 恒等于 | `\equiv` |  $\equiv$ |
| 同余 | `\pmod{n}` | `a \equiv b \pmod{n}` ->  $a \equiv b \pmod{n}$ |
| 属于 | `\in` | `a \in b` -> $a \in b$|
| 累加 | `\sum` | `\sum_{i=1}^{n} a_{i}` -> $\sum_{i=1}^{n} a_{i}$ |
| 累乘 | `\prod` | `\prod_{i=1}^{n} a_{i}` -> $\prod_{i=1}^{n} a_{i}$ |
| 整除 | `\mid` |`a \mid b` -> $a \mid b$ |
| 不能整除 | `\nmid` |`a \nmid b` -> $a \nmid b$ |
| 相似 | `\sim` | `a \sim b` -> $a \sim b$ |
| 向下取整 | `\lfloor\rfloor` | `$\lfloor\frac{a}{b}\rfloor$` -> $\lfloor\frac{a}{b}\rfloor$ |

**多行LaTeX公式对齐**

```LaTeX
$$
\begin{split}
    (a_{k}a_{k-1} \cdots a_{2} a_{1} a_{0})_{10}&= a_{k}10^{k} + a_{k-1}k^{k-1} + \cdots + a_{1}10 + a_{0} \\
    &\equiv a_{k} + a_{k-1} + \cdots + a_{1} + a_{0} \pmod{3}
\end{split}
$$
```
$$
\begin{split}
    (a_{k}a_{k-1} \cdots a_{2} a_{1} a_{0})_{10}&= a_{k}10^{k} + a_{k-1}k^{k-1} + \cdots + a_{1}10 + a_{0} \\
    &\equiv a_{k} + a_{k-1} + \cdots + a_{1} + a_{0} \pmod{3}
\end{split}
$$

**多项式**

```LaTex
$$
\gcd(a, b) = 
\begin{cases}
a, & b = 0 \\
\gcd \left(b, a \bmod b\right), & b \neq 0
\end{cases}
$$
```

$$
\gcd(a, b) = 
\begin{cases}
a, & b = 0 \\
\gcd \left(b, a \bmod b\right), & b \neq 0
\end{cases}
$$


### 大写希腊字母
|希腊字母|LaTeX表示|示例代码（在数学模式下，如`$`包裹）|显示效果|
| ---- | ---- | ---- | ---- |
|$\Alpha$|`\Alpha`|`$\Alpha$`|$\Alpha$|
|$\Beta$|`\Beta`|`$\Beta$`|$\Beta$|
|$\Gamma$|`\Gamma`|`$\Gamma$`|$\Gamma$|
|$\Delta$|`\Delta`|`$\Delta$`|$\Delta$|
|$\Epsilon$|`\Epsilon`|`$\Epsilon$`|$\Epsilon$|
|$\Zeta$|`\Zeta`|`$\Zeta$`|$\Zeta$|
|$\Eta$|`\Eta`|`$\Eta$`|$\Eta$|
|$\Theta$|`\Theta`|`$\Theta$`|$\Theta$|
|$\Iota$|`\Iota`|`$\Iota$`|$\Iota$|
|$\Kappa$|`\Kappa`|`$\Kappa$`|$\Kappa$|
|$\Lambda$|`\Lambda`|`$\Lambda$`|$\Lambda$|
|$\Mu$|`\Mu`|`$\Mu$`|$\Mu$|
|$\Nu$|`\Nu`|`$\Nu$`|$\Nu$|
|$\Xi$|`\Xi`|`$\Xi$`|$\Xi$|
|$\Pi$|`\Pi`|`$\Pi$`|$\Pi$|
|$\Rho$|`\Rho`|`$\Rho$`|$\Rho$|
|$\Sigma$|`\Sigma`|`$\Sigma$`|$\Sigma$|
|$\Tau$|`\Tau`|`$\Tau$`|$\Tau$|
|$\Upsilon$|`\Upsilon`|`$\Upsilon$`|$\Upsilon$|
|$\Phi$|`\Phi`|`$\Phi$`|$\Phi$|
|$\Chi$|`\Chi`|`$\Chi$`|$\Chi$|
|$\Psi$|`\Psi`|`$\Psi$`|$\Psi$|
|$\Omega$|`\Omega`|`$\Omega$`|$\Omega$|

### 小写希腊字母
|希腊字母|LaTeX表示|示例代码（在数学模式下，如`$`包裹）|显示效果|
| ---- | ---- | ---- | ---- |
|$\alpha$|`\alpha`|`$\alpha$`|$\alpha$|
|$\beta$|`\beta`|`$\beta$`|$\beta$|
|$\gamma$|`\gamma`|`$\gamma$`|$\gamma$|
|$\delta$|`\delta`|`$\delta$`|$\delta$|
|$\epsilon$|`\epsilon`|`$\epsilon$`|$\epsilon$|
|$\zeta$|`\zeta`|`$\zeta$`|$\zeta$|
|$\eta$|`\eta`|`$\eta$`|$\eta$|
|$\theta$|`\theta`|`$\theta$`|$\theta$|
|$\iota$|`\iota`|`$\iota$`|$\iota$|
|$\kappa$|`\kappa`|`$\kappa$`|$\kappa$|
|$\lambda$|`\lambda`|`$\lambda$`|$\lambda$|
|$\mu$|`\mu`|`$\mu$`|$\mu$|
|$\nu$|`\nu`|`$\nu$`|$\nu$|
|$\xi$|`\xi`|`$\xi$`|$\xi$|
|$\pi$|`\pi`|`$\pi$`|$\pi$|
|$\rho$|`\rho`|`$\rho$`|$\rho$|
|$\sigma$|`\sigma`|`$\sigma$`|$\sigma$|
|$\tau$|`\tau`|`$\tau$`|$\tau$|
|$\upsilon$|`\upsilon`|`$\upsilon$`|$\upsilon$|
|$\varphi$|`\varphi`|`$\varphi$`|$\varphi$|
|$\chi$|`\chi`|`$\chi$`|$\chi$|
|$\psi$|`\psi`|`$\psi$`|$\psi$|
|$\omega$|`\omega`|`$\omega$`|$\omega$| 