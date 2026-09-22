"""
utils.py - Utility functions for PVfit
"""

import math


def fast_erf(x):
    """Fast approximation of error function.

    Args:
        x (float): Input value

    Returns:
        float: Approximate error function value
    """
    mod_x = abs(x)
    p = (
        1
        + 0.278393 * mod_x
        + 0.230389 * (x * x)
        + 0.000972 * mod_x * (x * x)
        + 0.078108 * (x * x) * (x * x)
    )
    return (1 if x >= 0 else -1) * (1 - 1.0 / ((p * p) * (p * p)))


def gauss_integral(x):
    """Gaussian integral function (cumulative distribution).

    Args:
        x (float): Input value

    Returns:
        float: Gaussian integral from -inf to x
    """
    inv_sqrt2 = 1 / math.sqrt(2)  # ≈ 0.707106781
    return 0.5 * (1 + fast_erf(x * inv_sqrt2))


def integ_for_threshold_cut(h, z, t, d):
    """
    Calculate sum and maximum within a d×d region.

    Args:
        h: 2D histogram array
        z, t: Bin indices (center position)
        d: Region radius in bin units

    Returns:
        tuple: (sum_val, max_val)
    """
    sum_val = 0
    max_val = 0
    # Calculate region boundaries
    z_min = max(0, z - d)
    z_max = min(h.shape[0] - 1, z + d)
    t_min = max(0, t - d)
    t_max = min(h.shape[1] - 1, t + d)
    # Iterate over all points within the region
    for dz in range(z_min, z_max + 1):
        for dt in range(t_min, t_max + 1):
            val = h[dz][dt]
            sum_val += val
            max_val = max(max_val, val)
    return sum_val, max_val


def is_point_in_ellipse(point, center, zerr, terr, scale_factor=1.0):
    """
    Check if point is inside elliptical region.

    Args:
        point: (z, t) Point coordinates to check
        center: (z, t) Ellipse center coordinates
        zerr, terr: Error values defining ellipse semi-axes
        scale_factor: Ellipse scaling factor

    Returns:
        bool: True if point is inside ellipse
    """
    z, t = point
    cz, ct = center
    return (
        (z - cz) ** 2 / (zerr * scale_factor) ** 2
        + (t - ct) ** 2 / (terr * scale_factor) ** 2
    ) <= 1


def getTukeyWeight(trchi2, max_chi2):
    """Calculate Tukey weight for robust fitting.

    Args:
        trchi2 (float): Chi-squared value
        max_chi2 (float): Maximum chi-squared threshold

    Returns:
        float: Tukey weight (0 to 1)
    """
    return (1.0 - (trchi2 / max_chi2)) ** 2 if trchi2 < max_chi2 else 0.0
