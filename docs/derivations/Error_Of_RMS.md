# Uncertainty of a sample standard deviation

[Back to derivations](README.md)

## What the code uses

`pvfit.analysis.utils.std_with_error` reports the sample standard deviation
$s$ (with $n-1$ in the denominator) and an uncertainty that makes no Gaussian
assumption. With $m_4$ the fourth central moment of the sample,

$$
\operatorname{Var}(s^2)\approx\frac{1}{n}\left[m_4-\frac{n-3}{n-1}\,s^4\right],
\qquad
\operatorname{Var}(s)\approx\frac{\operatorname{Var}(s^2)}{4s^2},
$$

the second step being the first-order delta method for $s=\sqrt{s^2}$. For a
Gaussian sample $m_4=3\sigma^4$ and the expression reduces to the familiar
$\operatorname{SE}(s)\approx\sigma/\sqrt{2n}$ derived below; heavier tails
give a larger uncertainty than the Gaussian formula would.

The `resolution` analysis applies this to residuals rec − true inside a window
and reports the std as the resolution and the mean as the bias, each with its
own uncertainty ($s/\sqrt{n}$ for the mean). It does not report the RMS about
zero, which would combine the two: $\mathrm{RMS}^2 \approx s^2 + \text{bias}^2$.

## Gaussian limit

For independent samples $X_i \sim \mathcal{N}(0,\sigma^2)$ with known zero mean,
define the sample RMS as

$$
\hat{s}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}X_i^2}.
$$

The first-order delta method gives the large-sample approximation

$$
\operatorname{SE}(\hat{s})\approx\frac{\sigma}{\sqrt{2n}}.
$$

## Derivation

The normalized sum of squares follows a chi-square distribution:

$$
Y=\frac{1}{\sigma^2}\sum_{i=1}^{n}X_i^2\sim\chi_n^2,
\qquad \mathbb{E}[Y]=n,\qquad \operatorname{Var}(Y)=2n.
$$

Write $\hat{s}=g(Y)=\sigma\sqrt{Y/n}$. Expanding around $\mathbb{E}[Y]=n$ gives

$$
g(Y)\approx g(n)+g'(n)(Y-n),\qquad g'(n)=\frac{\sigma}{2n}.
$$

Hence

$$
\operatorname{Var}(\hat{s})\approx[g'(n)]^2\operatorname{Var}(Y)
=\frac{\sigma^2}{2n}.
$$

Taking the square root gives the stated standard error. This approximation
omits higher-order terms; it is not an exact finite-sample identity.

## Scope

If the mean is unknown and the estimator is the sample standard deviation
$s=\sqrt{\sum_i(X_i-\bar{X})^2/(n-1)}$, the same argument replaces $n$ by $n-1$:

$$
\operatorname{SE}(s)\approx\frac{\sigma}{\sqrt{2(n-1)}}.
$$

For non-Gaussian distributions, the delta-method calculation depends on the
fourth moment. The Gaussian coefficient does not apply in general.
