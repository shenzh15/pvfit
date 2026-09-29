# Derivations

[Back to README](../../README.md)

These notes and notebooks describe the mathematics behind the implementation.
The runtime package does not import them.

| File | Topic |
| --- | --- |
| [math_in_track_fit.ipynb](math_in_track_fit.ipynb) | residuals and derivatives of the track-vertex chi2 |
| [tukeyweight_chi2.ipynb](tukeyweight_chi2.ipynb) | Tukey weight behaviour as a function of chi2 |
| [zbl_uncertainty_derivation.ipynb](zbl_uncertainty_derivation.ipynb) | uncertainty on the beamline `z` intersection, two equivalent routes |
| [Gaussian_cdf_approximation.ipynb](Gaussian_cdf_approximation.ipynb) | the polynomial Gaussian-CDF approximation used in the seeding |
| [Error_Of_RMS.md](Error_Of_RMS.md) | uncertainty of a sample standard deviation: the fourth-moment formula the code uses, and its Gaussian limit `sigma / sqrt(2n)` |

From the project root, install the dev dependency group (`sympy`, `ipykernel`):

```bash
uv sync --group dev
```
