# Deriving the Standard Error of RMS to $\displaystyle \sigma/\sqrt{2n}$

> **Target Formula**  
> For $n$ independent and identically distributed samples from $\mathcal{N}(0,\sigma^2)$  
> $$
> \hat{s} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}X_i^{2}}
> $$  
> we have  
> $$
> \operatorname{SE}(\hat{s}) = \frac{\sigma}{\sqrt{2n}}
> $$

---

## 1. Definitions and Prerequisites

1. **Data Assumption**  
   $$
   X_i \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0,\sigma^2), \quad i=1,\ldots,n
   $$

2. **Sample RMS (Root Mean Square)**  
   $$
   \hat{s} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}X_i^{2}} \quad
   \text{(Since the mean is known to be 0, } \hat{s} \text{ is also called "sample standard deviation")}
   $$

---

## 2. Sum of Squares $\longrightarrow$ $\chi^2$ Distribution

- **Single Gaussian**  
  $$
  \frac{X_i^2}{\sigma^2} \sim \chi^2_{(1)}
  $$

- **n Independent Gaussians**  
  $$
  Y = \frac{1}{\sigma^{2}}\sum_{i=1}^{n}X_i^{2}
  \quad \Longrightarrow \quad Y \sim \chi^2_{(n)}
  $$

- **Rewrite** $\hat{s}$ **using Y**  
  $$
  \hat{s} = \sigma \sqrt{\frac{Y}{n}}
  $$

---

## 3. Two Fundamental Properties of $\chi^2$ Distribution

For a $\chi^2$ variable with degrees of freedom $k$:

$$
\boxed{
\mathbb{E}[Y]=k, \quad
\operatorname{Var}(Y)=2k
}
$$

Here $k=n$, so

$$
\mathbb{E}[Y]=n, \quad
\operatorname{Var}(Y)=2n
$$

---

## 4. Delta Method: Variance Approximation for Nonlinear Functions

1. **Let**  
   $$
   g(y) = \sigma \sqrt{\frac{y}{n}}
   $$

2. **First-order Taylor expansion around the mean** $y_0=\mathbb{E}[Y]=n$  
   $$
   g(Y) \approx g(y_0) + g'(y_0)(Y-y_0)
   $$

3. **Variance Propagation**  
   $$
   \operatorname{Var}[g(Y)] \approx
   \bigl(g'(y_0)\bigr)^2 \operatorname{Var}(Y)
   $$

---

## 5. Calculate Derivative and Substitute

- **Derivative**  
  $$
  g'(y) = \sigma \frac{1}{2}\frac{1}{\sqrt{ny}}
  $$

- **At** $y_0=n$  
  $$
  g'(y_0)=\sigma \frac{1}{2}\frac{1}{\sqrt{n \cdot n}}
           =\frac{\sigma}{2n}
  $$

- **Substitute back into variance**  
  $$
  \operatorname{Var}(\hat{s})
    \approx
    \left(\frac{\sigma}{2n}\right)^2 (2n)
    =\frac{\sigma^{2}}{2n}
  $$

- **Standard Error (square root of variance)**  
  $$
  \boxed{\operatorname{SE}(\hat{s})=\frac{\sigma}{\sqrt{2n}}}
  $$

---

## 6. About the Accuracy of the "Approximation"

- Including second-order terms gives  
  $$
  \operatorname{Var}(\hat{s})
  =\frac{\sigma^{2}}{2n}\left(1-\frac{1}{2n}+O(n^{-2})\right)
  $$
- When $n \geq 10$, the first-order approximation is already very close to the true value.

---

## 7. Summary and Key Points to Remember

| Step | Key Formula | Must Remember |
|------|-------------|---------------|
| **①** Gaussian sum of squares → $\chi^2$ | $Y = \frac{1}{\sigma^2}\sum X_i^2 \sim \chi^2_{(n)}$ | One of the most commonly used Gaussian properties |
| **②** $\chi^2$ mean and variance | $\mathbb{E}[Y]=n, \operatorname{Var}(Y)=2n$ | Memorize this |
| **③** Delta method | $\operatorname{Var}[g(Y)] \approx g'(\mathbb{E} Y)^2\operatorname{Var}(Y)$ | Master the approach |
| **④** Substitute to find SE | $\operatorname{SE}(\hat{s})=\sigma/\sqrt{2n}$ | The conclusion |

---

### ✎ Notes

1. **Unknown mean** case requires centering first and dividing by $(n-1)$, then  
   $$
   \operatorname{SE}(\hat{s})\approx\frac{\sigma}{\sqrt{2(n-1)}}
   $$
2. **Non-Gaussian distributions** can also use the Delta method for RMS, but the coefficient $\sigma/\sqrt{2}$ only holds for Gaussian distributions because it relies on the $\chi^2$ structure.
3. The Delta method's choice to **expand around** $\mathbb{E}[Y]$ **is the most natural and efficient—it centers the error terms and minimizes computation.**

> **One-sentence memory aid**  
> **"Sum of squares→χ², remember mean and variance, then apply Delta: three steps to get σ/√(2n)."**
