
**向下取整**
`\lfloor\rfloor`

\(\lfloor{\frac{a+c}{b+d}}\rfloor\)

> 通过`\left\right`自动调整花括号大小
> \(\left\lfloor{\frac{a+c}{b+d}}\right\rfloor\)

**分数**
`\frac{a}{b}`

\(\frac{a}{b}\)


**同余**
`\pmod{n}`
\(\pmod{n}\)

`\(a \equiv b \pmod{n}\)`
\(a \equiv b \pmod{n}\)

**空格**

| 命令         | 宽度               | 示例（语法 → 效果）                     |
|--------------|--------------------|----------------------------------------|
| `\,`         | 小空格（约 `1/6 em`） | `a\,b` → \(a\,b\)                     |
| `\:` 或 `\>` | 中等空格（约 `2/7 em`） | `a\:b` → \(a\:b\) 或 \(a\>b\)          |
| `\;`         | 大空格（约 `5/18 em`） | `a\;b` → \(a\;b\)                     |
| `\quad`      | 标准空格（`1 em`） | `a\quad b` → \(a\quad b\)             |
| `\qquad`     | 双倍空格（`2 em`） | `a\qquad b` → \(a\qquad b\)           |


| 常用字符| 公式 | 实例 |
| ------ | ---- | -------- |
| 乘号 | `\times` | `a \times b` -> \(a \times b\) |
| 点乘 | `\cdot` | `a \cdot b` -> \(a \cdot b\) |
| 恒等于 | `\equiv` |  \(\equiv\) |
| 同余 | `\pmod{n}` | `a \equiv b \pmod{n}` ->  \(a \equiv b \pmod{n}\) |
| 属于 | `\in` | `a \in b` -> \(a \in b\)|

**多行LaTeX公式对齐**

```LaTeX
\[
\begin{split}
    (a_{k}a_{k-1} \cdots a_{2} a_{1} a_{0})_{10}&= a_{k}10^{k} + a_{k-1}k^{k-1} + \cdots + a_{1}10 + a_{0} \\
    &\equiv a_{k} + a_{k-1} + \cdots + a_{1} + a_{0} \pmod{3}
\end{split}
\]
```
\[
\begin{split}
    (a_{k}a_{k-1} \cdots a_{2} a_{1} a_{0})_{10}&= a_{k}10^{k} + a_{k-1}k^{k-1} + \cdots + a_{1}10 + a_{0} \\
    &\equiv a_{k} + a_{k-1} + \cdots + a_{1} + a_{0} \pmod{3}
\end{split}
\]