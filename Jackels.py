# ++++++++++++++++++++++++++++++++++++++++++++++++++++s+++++++++++++++++++++++++++++++++++
# This code was translated from C++ into Python.
# The original C++ code resides at www.jaeckel.org/LetsBeRational.7z
#                       ***ALL ERRORS ARE OURS***
# ++++++++++++++++++++++++++++++++++++++++++#+++++++++++++++++++++++++++++++++++++++++++++

# ======================================================================================
# Copyright © 2013-2014 Peter Jäckel.
#
# Permission to use, copy, modify, and distribute this software is freely granted,
# provided that this notice is preserved.
#
# WARRANTY DISCLAIMER
# The Software is provided "as is" without warranty of any kind, either express or implied,
# including without limitation any implied warranties of condition, uninterrupted use,
# merchantability, fitness for a particular purpose, or non-infringement.
# ======================================================================================

TWO_PI = 6.283185307179586476925286766559005768394338798750
SQRT_PI_OVER_TWO = 1.253314137315500251207882642405522626503493370305  # sqrt(pi/2) to avoid misinterpretation.
SQRT_THREE = 1.732050807568877293527446341505872366942805253810
SQRT_ONE_OVER_THREE = 0.577350269189625764509148780501957455647601751270
TWO_PI_OVER_SQRT_TWENTY_SEVEN = 1.209199576156145233729385505094770488189377498728 # 2*pi/sqrt(27)
PI_OVER_SIX = 0.523598775598298873077107230546583814032861566563

ONE_OVER_SQRT_TWO = 0.7071067811865475244008443621048490392848359376887
ONE_OVER_SQRT_TWO_PI = 0.3989422804014326779399460599343818684758586311649
SQRT_TWO_PI = 2.506628274631000502415765284811045253006986740610

import sys
from math import sqrt, log, exp, isnan


DBL_MIN = sys.float_info.min
DBL_MAX = sys.float_info.max
DBL_EPSILON = sys.float_info.epsilon
SQRT_DBL_EPSILON = sqrt(sys.float_info.epsilon);
FOURTH_ROOT_DBL_EPSILON = sqrt(SQRT_DBL_EPSILON);
EIGHTH_ROOT_DBL_EPSILON = sqrt(FOURTH_ROOT_DBL_EPSILON);
SIXTEENTH_ROOT_DBL_EPSILON = sqrt(EIGHTH_ROOT_DBL_EPSILON);
SQRT_DBL_MIN = sqrt(sys.float_info.min);
SQRT_DBL_MAX = sqrt(sys.float_info.max);

# Set this to 0 if you want positive results for (positive) denormalized inputs, else to DBL_MIN.
# Note that you cannot achieve full machine accuracy from denormalized inputs!
DENORMALIZATION_CUTOFF = 0;

VOLATILITY_VALUE_TO_SIGNAL_PRICE_IS_BELOW_INTRINSIC = -sys.float_info.max;
VOLATILITY_VALUE_TO_SIGNAL_PRICE_IS_ABOVE_MAXIMUM = sys.float_info.max;

def is_below_horizon(x): return abs(x) < DENORMALIZATION_CUTOFF; # This weeds out denormalized (a.k.a. 'subnormal') numbers.

from scipy import expm1


def norm_pdf(x): return ONE_OVER_SQRT_TWO_PI*exp(-.5*x*x);

def normalised_intrinsic(x, q): #q=±1
    if (q*x<=0):
        return 0;
    else:
        return q * (expm1(-0.5*x)+1) * expm1(x)

def normalised_intrinsic_call(x): return normalised_intrinsic(x,1)

from scipy.special import erf, erfc, erfcx, ndtr, ndtri


def asymptotic_expansion_of_normalized_black_call(h, t):
    e = (t / h) * (t / h)
    r = ((h + t) * (h - t))
    q = (h / r) * (h / r)
    # 17th order asymptotic expansion of A(h,t) in q, sufficient for F(h) [and thus y(h)] to have relative accuracy of 1.64E-16 for h <= ?  with  ?:=-10.
    asymptotic_expansion_sum = (2.0 + q * (-6.0E0 - 2.0 * e
                                           + 3.0 * q * (1.0E1 + e * (2.0E1 + 2.0 * e)
                                                        + 5.0 * q * (-1.4E1 + e * (-7.0E1 + e * (-4.2E1 - 2.0 * e))
                                                                     + 7.0 * q * (1.8E1 + e * (
    1.68E2 + e * (2.52E2 + e * (7.2E1 + 2.0 * e)))
                                                                                  + 9.0 * q * (-2.2E1 + e * (
    -3.3E2 + e * (-9.24E2 + e * (-6.6E2 + e * (-1.1E2 - 2.0 * e))))
                                                                                               + 1.1E1 * q * (
                                                                                               2.6E1 + e * (
                                                                                               5.72E2 + e * (
                                                                                               2.574E3 + e * (
                                                                                               3.432E3 + e * (
                                                                                               1.43E3 + e * (
                                                                                               1.56E2 + 2.0 * e)))))
                                                                                               + 1.3E1 * q * (
                                                                                               -3.0E1 + e * (
                                                                                               -9.1E2 + e * (
                                                                                               -6.006E3 + e * (
                                                                                               -1.287E4 + e * (
                                                                                               -1.001E4 + e * (
                                                                                               -2.73E3 + e * (
                                                                                               -2.1E2 - 2.0 * e))))))
                                                                                               + 1.5E1 * q * (
                                                                                               3.4E1 + e * (
                                                                                               1.36E3 + e * (
                                                                                               1.2376E4 + e * (
                                                                                               3.8896E4 + e * (
                                                                                               4.862E4 + e * (
                                                                                               2.4752E4 + e * (
                                                                                               4.76E3 + e * (
                                                                                               2.72E2 + 2.0 * e)))))))
                                                                                               + 1.7E1 * q * (
                                                                                               -3.8E1 + e * (
                                                                                               -1.938E3 + e * (
                                                                                               -2.3256E4 + e * (
                                                                                               -1.00776E5 + e * (
                                                                                               -1.84756E5 + e * (
                                                                                               -1.51164E5 + e * (
                                                                                               -5.4264E4 + e * (
                                                                                               -7.752E3 + e * (
                                                                                               -3.42E2 - 2.0 * e))))))))
                                                                                               + 1.9E1 * q * (
                                                                                               4.2E1 + e * (
                                                                                               2.66E3 + e * (
                                                                                               4.0698E4 + e * (
                                                                                               2.3256E5 + e * (
                                                                                               5.8786E5 + e * (
                                                                                               7.05432E5 + e * (
                                                                                               4.0698E5 + e * (
                                                                                               1.08528E5 + e * (
                                                                                               1.197E4 + e * (
                                                                                               4.2E2 + 2.0 * e)))))))))
                                                                                               + 2.1E1 * q * (
                                                                                               -4.6E1 + e * (
                                                                                               -3.542E3 + e * (
                                                                                               -6.7298E4 + e * (
                                                                                               -4.90314E5 + e * (
                                                                                               -1.63438E6 + e * (
                                                                                               -2.704156E6 + e * (
                                                                                               -2.288132E6 + e * (
                                                                                               -9.80628E5 + e * (
                                                                                               -2.01894E5 + e * (
                                                                                               -1.771E4 + e * (
                                                                                               -5.06E2 - 2.0 * e))))))))))
                                                                                               + 2.3E1 * q * (
                                                                                               5.0E1 + e * (
                                                                                               4.6E3 + e * (
                                                                                               1.0626E5 + e * (
                                                                                               9.614E5 + e * (
                                                                                               4.08595E6 + e * (
                                                                                               8.9148E6 + e * (
                                                                                               1.04006E7 + e * (
                                                                                               6.53752E6 + e * (
                                                                                               2.16315E6 + e * (
                                                                                               3.542E5 + e * (
                                                                                               2.53E4 + e * (
                                                                                               6.0E2 + 2.0 * e)))))))))))
                                                                                               + 2.5E1 * q * (
                                                                                               -5.4E1 + e * (
                                                                                               -5.85E3 + e * (
                                                                                               -1.6146E5 + e * (
                                                                                               -1.77606E6 + e * (
                                                                                               -9.37365E6 + e * (
                                                                                               -2.607579E7 + e * (
                                                                                               -4.01166E7 + e * (
                                                                                               -3.476772E7 + e * (
                                                                                               -1.687257E7 + e * (
                                                                                               -4.44015E6 + e * (
                                                                                               -5.9202E5 + e * (
                                                                                               -3.51E4 + e * (
                                                                                               -7.02E2 - 2.0 * e))))))))))))
                                                                                               + 2.7E1 * q * (
                                                                                               5.8E1 + e * (
                                                                                               7.308E3 + e * (
                                                                                               2.3751E5 + e * (
                                                                                               3.12156E6 + e * (
                                                                                               2.003001E7 + e * (
                                                                                               6.919458E7 + e * (
                                                                                               1.3572783E8 + e * (
                                                                                               1.5511752E8 + e * (
                                                                                               1.0379187E8 + e * (
                                                                                               4.006002E7 + e * (
                                                                                               8.58429E6 + e * (
                                                                                               9.5004E5 + e * (
                                                                                               4.7502E4 + e * (
                                                                                               8.12E2 + 2.0 * e)))))))))))))
                                                                                               + 2.9E1 * q * (
                                                                                               -6.2E1 + e * (
                                                                                               -8.99E3 + e * (
                                                                                               -3.39822E5 + e * (
                                                                                               -5.25915E6 + e * (
                                                                                               -4.032015E7 + e * (
                                                                                               -1.6934463E8 + e * (
                                                                                               -4.1250615E8 + e * (
                                                                                               -6.0108039E8 + e * (
                                                                                               -5.3036505E8 + e * (
                                                                                               -2.8224105E8 + e * (
                                                                                               -8.870433E7 + e * (
                                                                                               -1.577745E7 + e * (
                                                                                               -1.472562E6 + e * (
                                                                                               -6.293E4 + e * (
                                                                                               -9.3E2 - 2.0 * e))))))))))))))
                                                                                               + 3.1E1 * q * (
                                                                                               6.6E1 + e * (
                                                                                               1.0912E4 + e * (
                                                                                               4.74672E5 + e * (
                                                                                               8.544096E6 + e * (
                                                                                               7.71342E7 + e * (
                                                                                               3.8707344E8 + e * (
                                                                                               1.14633288E9 + e * (
                                                                                               2.07431664E9 + e * (
                                                                                               2.33360622E9 + e * (
                                                                                               1.6376184E9 + e * (
                                                                                               7.0963464E8 + e * (
                                                                                               1.8512208E8 + e * (
                                                                                               2.7768312E7 + e * (
                                                                                               2.215136E6 + e * (
                                                                                               8.184E4 + e * (
                                                                                               1.056E3 + 2.0 * e)))))))))))))))
                                                                                               + 3.3E1 * (-7.0E1 + e * (
                                                                                               -1.309E4 + e * (
                                                                                               -6.49264E5 + e * (
                                                                                               -1.344904E7 + e * (
                                                                                               -1.4121492E8 + e * (
                                                                                               -8.344518E8 + e * (
                                                                                               -2.9526756E9 + e * (
                                                                                               -6.49588632E9 + e * (
                                                                                               -9.0751353E9 + e * (
                                                                                               -8.1198579E9 + e * (
                                                                                               -4.6399188E9 + e * (
                                                                                               -1.6689036E9 + e * (
                                                                                               -3.67158792E8 + e * (
                                                                                               -4.707164E7 + e * (
                                                                                               -3.24632E6 + e * (
                                                                                               -1.0472E5 + e * (
                                                                                               -1.19E3 - 2.0 * e))))))))))))))))) * q)))))))))))))))));
    b = ONE_OVER_SQRT_TWO_PI * exp((-0.5 * (h * h + t * t))) * (t / r) * asymptotic_expansion_sum
    return abs(max(b, 0.0))


def normalised_black_call_using_erfcx(h, t):
    b = 0.5 * exp(-0.5 * (h * h + t * t)) * (erfcx(-ONE_OVER_SQRT_TWO * (h + t)) - erfcx(-ONE_OVER_SQRT_TWO * (h - t)));
    return abs(max(b, 0.0))


def small_t_expansion_of_normalized_black_call(h, t):
    # Y(h) := F(h)/f(h) = v(p/2)·erfcx(-h/v2)
    # a := 1+h·Y(h)  --- Note that due to h<0, and h·Y(h) -> -1 (from above) as h -> -8, we also have that a>0 and a -> 0 as h -> -8
    # w := t² , h2 := h²
    a = 1 + h * (0.5 * SQRT_TWO_PI) * erfcx(-ONE_OVER_SQRT_TWO * h)
    w = t * t
    h2 = h * h
    expansion = 2 * t * (a + w * ((-1 + 3 * a + a * h2) / 6 + w * ((-7 + 15 * a + h2 * (-1 + 10 * a + a * h2)) / 120
                                                                   + w * ((-57 + 105 * a + h2 * (
    -18 + 105 * a + h2 * (-1 + 21 * a + a * h2))) / 5040
                                                                          + w * ((-561 + 945 * a + h2 * (
    -285 + 1260 * a + h2 * (-33 + 378 * a + h2 * (-1 + 36 * a + a * h2)))) / 362880
                                                                                 + w * ((-6555 + 10395 * a + h2 * (
    -4680 + 17325 * a + h2 * (-840 + 6930 * a + h2 * (-52 + 990 * a + h2 * (-1 + 55 * a + a * h2))))) / 39916800
                                                                                        + ((-89055 + 135135 * a + h2 * (
    -82845 + 270270 * a + h2 * (-20370 + 135135 * a + h2 * (
    -1926 + 25740 * a + h2 * (-75 + 2145 * a + h2 * (-1 + 78 * a + a * h2)))))) * w) / 6227020800.0))))))
    b = ONE_OVER_SQRT_TWO_PI * exp((-0.5 * (h * h + t * t))) * expansion
    return abs(max(b, 0.0))


def normalized_black_call_using_norm_cdf(x, s):
    h = x / s
    t = 0.5 * s
    b_max = exp(0.5 * x)

    b = ndtr(h + t) * b_max - ndtr(h - t) / b_max
    return abs(max(b, 0.0))


asymptotic_expansion_accuracy_threshold = -10
small_t_expansion_of_normalized_black_threshold = 2 * SIXTEENTH_ROOT_DBL_EPSILON


def normalised_black_call(x, s):
    if (x > 0): return normalised_intrinsic_call(x) + normalised_black_call(-x, s)
    ax = abs(x);

    if (s <= ax * DENORMALIZATION_CUTOFF): return normalised_intrinsic_call(x)
    # Denote h := x/s and t := s/2.
    # We evaluate the condition |h|>|?|, i.e., h<?  &&  t < t+|h|-|?|  avoiding any divisions by s , where ? = asymptotic_expansion_accuracy_threshold  and t = small_t_expansion_of_normalized_black_threshold .
    if (x < s * asymptotic_expansion_accuracy_threshold and 0.5 * s * s + x < s * (
        small_t_expansion_of_normalized_black_threshold + asymptotic_expansion_accuracy_threshold)):
        # Region 1.
        return asymptotic_expansion_of_normalized_black_call(x / s, 0.5 * s)
    if (0.5 * s < small_t_expansion_of_normalized_black_threshold):
        # Region 2.
        return small_t_expansion_of_normalized_black_call(x / s, 0.5 * s)
    # When b is more than, say, about 85% of b_max=exp(x/2), then b is dominated by the first of the two terms in the Black formula, and we retain more accuracy by not attempting to combine the two terms in any way.
    # We evaluate the condition h+t>0.85  avoiding any divisions by s.
    if (x + 0.5 * s * s > s * 0.85):
        # Region 3.
        return normalized_black_call_using_norm_cdf(x, s)
    # Region 4.
    return normalised_black_call_using_erfcx(x / s, 0.5 * s)


def square(x): return x * x


def normalised_vega(x, s):
    ax = abs(x);
    if ax <= 0:
        return ONE_OVER_SQRT_TWO_PI * exp(-0.125 * s * s)
    else:
        return 0 if (s <= 0 or s <= ax * SQRT_DBL_MIN) else  ONE_OVER_SQRT_TWO_PI * exp(
            -0.5 * (square(x / s) + square(0.5 * s)))


def normalised_black(x, s, q):  # q=±1
    return normalised_black_call(-x if q < 0 else x, s);  # Reciprocal-strike call-put equivalence


def black(F, K, sigma, T, q):  # q=±1
    intrinsic = abs(max((K - F if q < 0 else F - K), 0.0));

    # Map in-the-money to out-of-the-money
    if (q * (F - K) > 0):
        return intrinsic + black(F, K, sigma, T, -q);

    return max(intrinsic, (sqrt(F) * sqrt(K)) * normalised_black(log(F / K), sigma * sqrt(T), q));


def BlackScholesPriceJaeckelRational(F, K, T, vol, DF=1.0, q=1.0):
    return DF * black(F, K, vol, T, q)

minimum_rational_cubic_control_parameter_value = -(1.0 - sqrt(DBL_EPSILON));
maximum_rational_cubic_control_parameter_value = 2.0 / (DBL_EPSILON * DBL_EPSILON);

def is_zero(x): return abs(x) < DBL_MIN


def rational_cubic_interpolation(x, x_l, x_r, y_l, y_r, d_l, d_r, r):
    h = (x_r - x_l);
    if (abs(h) <= 0): return 0.5 * (y_l + y_r)

    # r should be greater than -1. We do not use  assert(r > -1)  here in order to allow values such as NaN to be propagated as they should.
    t = (x - x_l) / h

    if (not (r >= maximum_rational_cubic_control_parameter_value)):
        t = (x - x_l) / h
        omt = 1 - t
        t2 = t * t
        omt2 = omt * omt

        # Formula (2.4) divided by formula (2.5)
        return (y_r * t2 * t + (r * y_r - h * d_r) * t2 * omt + (r * y_l + h * d_l) * t * omt2 + y_l * omt2 * omt) / (
        1 + (r - 3) * t * omt)

    # Linear interpolation without over-or underflow.
    return y_r * t + y_l * (1 - t)


def rational_cubic_control_parameter_to_fit_second_derivative_at_left_side(x_l, x_r, y_l, y_r, d_l, d_r,
                                                                           second_derivative_l):
    h = (x_r - x_l)
    numerator = 0.5 * h * second_derivative_l + (d_r - d_l)

    if is_zero(numerator): return 0

    denominator = (y_r - y_l) / h - d_l

    if (is_zero(denominator)):
        return maximum_rational_cubic_control_parameter_value if numerator > 0 else minimum_rational_cubic_control_parameter_value;

    return numerator / denominator


def rational_cubic_control_parameter_to_fit_second_derivative_at_right_side(x_l, x_r, y_l, y_r, d_l, d_r,
                                                                            second_derivative_r):
    h = (x_r - x_l)
    numerator = 0.5 * h * second_derivative_r + (d_r - d_l)

    if is_zero(numerator): return 0

    denominator = d_r - (y_r - y_l) / h
    if (is_zero(denominator)):
        return maximum_rational_cubic_control_parameter_value if numerator > 0 else minimum_rational_cubic_control_parameter_value;

    return numerator / denominator


def minimum_rational_cubic_control_parameter(d_l, d_r, s, preferShapePreservationOverSmoothness):
    monotonic = d_l * s >= 0 and d_r * s >= 0
    convex = d_l <= s and s <= d_r
    concave = d_l >= s and s >= d_r;

    # If 3==r_non_shape_preserving_target, this means revert to standard cubic.
    if (not monotonic and not convex and not concave):
        return minimum_rational_cubic_control_parameter_value

    d_r_m_d_l = d_r - d_l
    d_r_m_s = d_r - s
    s_m_d_l = s - d_l
    r1 = -DBL_MAX
    r2 = r1

    # If monotonicity on this interval is possible, set r1 to satisfy the monotonicity condition (3.8).
    if (monotonic):
        if (not is_zero(s)):  # (3.8), avoiding division by zero.
            r1 = (d_r + d_l) / s  # (3.8)
        elif (
        preferShapePreservationOverSmoothness):  # If division by zero would occur, and shape preservation is preferred, set value to enforce linear interpolation.
            r1 = maximum_rational_cubic_control_parameter_value  # This value enforces linear interpolation.

    if (convex or concave):
        if (not (is_zero(s_m_d_l) or is_zero(d_r_m_s))):  # (3.18), avoiding division by zero.
            r2 = max(abs(d_r_m_d_l / d_r_m_s), abs(d_r_m_d_l / s_m_d_l));
        elif (preferShapePreservationOverSmoothness):
            r2 = maximum_rational_cubic_control_parameter_value  # This value enforces linear interpolation.

    elif (monotonic and preferShapePreservationOverSmoothness):
        r2 = maximum_rational_cubic_control_parameter_value  # This enforces linear interpolation along segments that are inconsistent with the slopes on the boundaries, e.g., a perfectly horizontal segment that has negative slopes on either edge.

    return max(minimum_rational_cubic_control_parameter_value, max(r1, r2))


def convex_rational_cubic_control_parameter_to_fit_second_derivative_at_left_side(x_l, x_r, y_l, y_r, d_l, d_r,
                                                                                  second_derivative_l,
                                                                                  preferShapePreservationOverSmoothness):
    r = rational_cubic_control_parameter_to_fit_second_derivative_at_left_side(x_l, x_r, y_l, y_r, d_l, d_r,
                                                                               second_derivative_l)
    r_min = minimum_rational_cubic_control_parameter(d_l, d_r, (y_r - y_l) / (x_r - x_l),
                                                     preferShapePreservationOverSmoothness)

    return max(r, r_min)


def convex_rational_cubic_control_parameter_to_fit_second_derivative_at_right_side(x_l, x_r, y_l, y_r, d_l, d_r,
                                                                                   second_derivative_r,
                                                                                   preferShapePreservationOverSmoothness):
    r = rational_cubic_control_parameter_to_fit_second_derivative_at_right_side(x_l, x_r, y_l, y_r, d_l, d_r,
                                                                                second_derivative_r)
    r_min = minimum_rational_cubic_control_parameter(d_l, d_r, (y_r - y_l) / (x_r - x_l),
                                                     preferShapePreservationOverSmoothness)

    return max(r, r_min)


def compute_f_lower_map_and_first_two_derivatives(x, s):
    ax = abs(x)
    z = SQRT_ONE_OVER_THREE * ax / s
    y = z * z
    s2 = s * s
    Phi = ndtr(-z)
    phi = ndtr(z)

    fpp = PI_OVER_SIX * y / (s2 * s) * Phi * (
    8 * SQRT_THREE * s * ax + (3 * s2 * (s2 - 8) - 8 * x * x) * Phi / phi) * exp(2 * y + 0.25 * s2)
    if (is_below_horizon(s)):
        fp = 1
        f = 0
    else:
        Phi2 = Phi * Phi
        fp = TWO_PI * y * Phi2 * exp(y + 0.125 * s * s)
        if (is_below_horizon(x)):
            f = 0
        else:
            f = TWO_PI_OVER_SQRT_TWENTY_SEVEN * ax * (Phi2 * Phi)

    return f, fp, fpp

def inverse_f_lower_map(x, f):
    if is_below_horizon(f):
        return 0
    else:
        return abs(x/(SQRT_THREE*ndtri((f/(TWO_PI_OVER_SQRT_TWENTY_SEVEN*abs(x)))**(1./3.))))

def compute_f_upper_map_and_first_two_derivatives(x, s):
    f = ndtr(-0.5 * s)
    if (is_below_horizon(x)):
        fp = -0.5
        fpp = 0
    else:
        w = square(x / s)
        fp = -0.5 * exp(0.5 * w)
        fpp = SQRT_PI_OVER_TWO * exp(w + 0.125 * s * s) * w / s

    return f, fp, fpp

def inverse_f_upper_map(f):
    return -2. * ndtri(f)

def householder_factor(newton, halley, hh3):
    return (1+0.5*halley*newton)/(1+newton*(halley+hh3*newton/6))


def unchecked_normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(beta, x, q, N):
    # Subtract intrinsic.
    if (q * x > 0):
        beta = abs(max(beta - normalised_intrinsic(x, q), 0.));
        q = -q;

    # Map puts to calls
    if (q < 0):
        x = -x;
        q = -q;

    if (beta <= 0):  # For negative or zero prices we return 0.
        return 0, 0;
    if (
        beta < DENORMALIZATION_CUTOFF):  # For positive but denormalized (a.k.a. 'subnormal') prices, we return 0 since it would be impossible to converge to full machine accuracy anyway.
        return 0, 0;
    b_max = exp(0.5 * x);
    if (beta >= b_max):
        return 0, VOLATILITY_VALUE_TO_SIGNAL_PRICE_IS_ABOVE_MAXIMUM;

    iterations = 0;
    direction_reversal_count = 0;
    f = -DBL_MAX;
    s = -DBL_MAX;
    ds = s;
    ds_previous = 0;
    s_left = DBL_MIN;
    s_right = DBL_MAX;
    # The temptation is great to use the optimised form b_c = exp(x/2)/2-exp(-x/2)·Phi(sqrt(-2·x)) but that would require implementing all of the above types of round-off and over/underflow handling for this expression, too.
    s_c = sqrt(abs(2 * x));
    b_c = normalised_black_call(x, s_c);
    v_c = normalised_vega(x, s_c);
    # Four branches.
    if (beta < b_c):
        s_l = s_c - b_c / v_c;
        b_l = normalised_black_call(x, s_l);
        if (beta < b_l):
            f_lower_map_l, d_f_lower_map_l_d_beta, d2_f_lower_map_l_d_beta2 = compute_f_lower_map_and_first_two_derivatives(
                x, s_l);
            r_ll = convex_rational_cubic_control_parameter_to_fit_second_derivative_at_right_side(0., b_l, 0.,
                                                                                                  f_lower_map_l, 1.,
                                                                                                  d_f_lower_map_l_d_beta,
                                                                                                  d2_f_lower_map_l_d_beta2,
                                                                                                  True);
            f = rational_cubic_interpolation(beta, 0., b_l, 0., f_lower_map_l, 1., d_f_lower_map_l_d_beta, r_ll);
            if (not (f > 0)):  # This can happen due to roundoff truncation for extreme values such as |x|>500.
                # We switch to quadratic interpolation using f(0)=0, f(b_l), and f'(0)=1 to specify the quadratic.
                t = beta / b_l;
                f = (f_lower_map_l * t + b_l * (1 - t)) * t;

            s = inverse_f_lower_map(x, f);
            s_right = s_l;

            while (iterations < N and abs(ds) > DBL_EPSILON * s):
                if (ds * ds_previous < 0):
                    direction_reversal_count += 1;
                if (iterations > 0 and (3 == direction_reversal_count or not (s > s_left and s < s_right))):
                    # If looping inefficently, or the forecast step takes us outside the bracket, or onto its edges, switch to binary nesting.
                    # NOTE that this can only really happen for very extreme values of |x|, such as |x|=|ln(F/K)| > 500.
                    s = 0.5 * (s_left + s_right);
                    if (s_right - s_left <= DBL_EPSILON * s): break;
                    direction_reversal_count = 0;
                    ds = 0;

                ds_previous = ds;
                b = normalised_black_call(x, s);
                bp = normalised_vega(x, s);
                if (b > beta and s < s_right):
                    s_right = s;
                elif (b < beta and s > s_left):
                    s_left = s;  # Tighten the bracket if applicable.
                if (b <= 0 or bp <= 0):  # Numerical underflow. Switch to binary nesting for this iteration.
                    ds = 0.5 * (s_left + s_right) - s;
                else:
                    ln_b = log(b);
                    ln_beta = log(beta);
                    bpob = bp / b;
                    h = x / s;
                    b_halley = h * h / s - s / 4;
                    newton = (ln_beta - ln_b) * ln_b / ln_beta / bpob;
                    halley = b_halley - bpob * (1 + 2 / ln_b);
                    b_hh3 = b_halley * b_halley - 3 * square(h / s) - 0.25;
                    hh3 = b_hh3 + 2 * square(bpob) * (1 + 3 / ln_b * (1 + 1 / ln_b)) - 3 * b_halley * bpob * (
                    1 + 2 / ln_b);
                    ds = newton * householder_factor(newton, halley, hh3);

                ds = max(-0.5 * s, ds);
                s += ds
                iterations += 1

            return iterations, s

        else:
            v_l = normalised_vega(x, s_l);
            r_lm = convex_rational_cubic_control_parameter_to_fit_second_derivative_at_right_side(b_l, b_c, s_l, s_c,
                                                                                                  1 / v_l, 1 / v_c, 0.0,
                                                                                                  False);
            s = rational_cubic_interpolation(beta, b_l, b_c, s_l, s_c, 1 / v_l, 1 / v_c, r_lm);
            s_left = s_l;
            s_right = s_c;


    else:
        s_h = s_c + (b_max - b_c) / v_c if v_c > DBL_MIN else s_c
        b_h = normalised_black_call(x, s_h);
        if (beta <= b_h):
            v_h = normalised_vega(x, s_h);
            r_hm = convex_rational_cubic_control_parameter_to_fit_second_derivative_at_left_side(b_c, b_h, s_c, s_h,
                                                                                                 1 / v_c, 1 / v_h, 0.0,
                                                                                                 False);
            s = rational_cubic_interpolation(beta, b_c, b_h, s_c, s_h, 1 / v_c, 1 / v_h, r_hm);
            s_left = s_c;
            s_right = s_h;

        else:
            f_upper_map_h, d_f_upper_map_h_d_beta, d2_f_upper_map_h_d_beta2 = compute_f_upper_map_and_first_two_derivatives(
                x, s_h);
            if (d2_f_upper_map_h_d_beta2 > -SQRT_DBL_MAX and d2_f_upper_map_h_d_beta2 < SQRT_DBL_MAX):
                r_hh = convex_rational_cubic_control_parameter_to_fit_second_derivative_at_left_side(b_h, b_max,
                                                                                                     f_upper_map_h, 0.,
                                                                                                     d_f_upper_map_h_d_beta,
                                                                                                     -0.5,
                                                                                                     d2_f_upper_map_h_d_beta2,
                                                                                                     True);
                f = rational_cubic_interpolation(beta, b_h, b_max, f_upper_map_h, 0., d_f_upper_map_h_d_beta, -0.5,
                                                 r_hh);

            if (f <= 0):
                h = b_max - b_h;
                t = (beta - b_h) / h
                f = (f_upper_map_h * (1 - t) + 0.5 * h * t) * (
                1 - t)  # We switch to quadratic interpolation using f(b_h), f(b_max)=0, and f'(b_max)=-1/2 to specify the quadratic.

            s = inverse_f_upper_map(f)
            s_left = s_h
            if (beta > 0.5 * b_max):  # Else we better drop through and let the objective function be g(s)=b(x,s)-beta.

                while (iterations < N and abs(ds) > DBL_EPSILON * s):
                    if (ds * ds_previous < 0): direction_reversal_count += 1;
                    if (iterations > 0 and (3 == direction_reversal_count or not (s > s_left and s < s_right))):
                        # If looping inefficently, or the forecast step takes us outside the bracket, or onto its edges, switch to binary nesting.
                        # NOTE that this can only really happen for very extreme values of |x|, such as |x|=|ln(F/K)| > 500.
                        s = 0.5 * (s_left + s_right);
                        if (s_right - s_left <= DBL_EPSILON * s): break;
                        direction_reversal_count = 0;
                        ds = 0;

                    ds_previous = ds;
                    b = normalised_black_call(x, s);
                    bp = normalised_vega(x, s);
                    if (b > beta and s < s_right):
                        s_right = s;
                    elif (b < beta and s > s_left):
                        s_left = s;  # Tighten the bracket if applicable.
                    if (
                            b >= b_max or bp <= DBL_MIN):  # Numerical underflow. Switch to binary nesting for this iteration.
                        ds = 0.5 * (s_left + s_right) - s;
                    else:
                        b_max_minus_b = b_max - b;
                        g = log((b_max - beta) / b_max_minus_b);
                        gp = bp / b_max_minus_b;
                        b_halley = square(x / s) / s - s / 4;
                        b_hh3 = b_halley * b_halley - 3 * square(x / (s * s)) - 0.25;
                        newton = -g / gp;
                        halley = b_halley + gp;
                        hh3 = b_hh3 + gp * (2 * gp + 3 * b_halley);
                        ds = newton * householder_factor(newton, halley, hh3);

                    ds = max(-0.5 * s, ds);
                    s += ds
                    iterations += 1

                return iterations, s

    while (iterations < N and abs(ds) > DBL_EPSILON * s):
        if (ds * ds_previous < 0): direction_reversal_count += 1;
        if (iterations > 0 and (3 == direction_reversal_count or not (s > s_left and s < s_right))):
            # If looping inefficently, or the forecast step takes us outside the bracket, or onto its edges, switch to binary nesting.
            # NOTE that this can only really happen for very extreme values of |x|, such as |x|=|ln(F/K)| > 500.
            s = 0.5 * (s_left + s_right);
            if (s_right - s_left <= DBL_EPSILON * s): break;
            direction_reversal_count = 0;
            ds = 0;

        ds_previous = ds;
        b = normalised_black_call(x, s);
        bp = normalised_vega(x, s);
        if (b > beta and s < s_right):
            s_right = s;
        elif (b < beta and s > s_left):
            s_left = s;  # Tighten the bracket if applicable.
        newton = (beta - b) / bp;
        halley = square(x / s) / s - s / 4;
        hh3 = halley * halley - 3 * square(x / (s * s)) - 0.25;
        ds = max(-0.5 * s, newton * householder_factor(newton, halley, hh3));
        s += ds
        iterations += 1

    return iterations, s


def normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(beta, x, q, N):
    # Map in-the-money to out-of-the-money
    if (q * x > 0):
        beta -= normalised_intrinsic(x, q);
        q = -q;

    if (beta < 0):
        return 0, VOLATILITY_VALUE_TO_SIGNAL_PRICE_IS_BELOW_INTRINSIC;
    return unchecked_normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(beta, x, q,
                                                                                                             N);

def normalised_implied_volatility_from_a_transformed_rational_guess(beta, x, q):
    return normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(beta,x,q,N=2);


def implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(price, F, K, T, q, N):
    intrinsic = abs(max((K - F if q < 0 else F - K), 0.0))
    if (price < intrinsic):
        return 0, VOLATILITY_VALUE_TO_SIGNAL_PRICE_IS_BELOW_INTRINSIC
    max_price = (K if q < 0 else F)
    if (price >= max_price):
        return 0, VOLATILITY_VALUE_TO_SIGNAL_PRICE_IS_ABOVE_MAXIMUM
    x = log(F / K)
    # Map in-the-money to out-of-the-money
    if (q * x > 0):
        price = abs(max(price - intrinsic, 0.0))
        q = -q

    return unchecked_normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(
        price / (sqrt(F) * sqrt(K)), x, q, N) / sqrt(T);

def implied_volatility_from_a_transformed_rational_guess(price, F, K, T, q):
    return implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(price,F,K,T,q, N=2)

def JaeckelRational(x, beta, q, relTol, N):

    nbrIter, sFound = normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(beta,\
                                                                                                                  x,q,N)
    return sFound, nbrIter


def isArbitrageFreePrice(F, K, p, DF=1.0, q=1.0, printInfo=False):
    """ Check if the input price is free of arbitrage.
        Print a warning message if it is not the case.

        Inputs:

            @F: the underlyer value
            @K: the strike
            @p: the input price

            @DF: the discount factor (default to 1.0)
            @q: 1.0 for a Call, -1.0 for a Put (default to 1.0)
            @printInfo: if True the output will print the for which combination of
                        p,K and F there is arbitrage

        Output:

            True if the price is arbitrage free, False otherwise.
    """
    if q == 1.0:

        inRange = DF * max(F - K, 0.0) < p < DF * F

    elif q == -1.0:

        inRange = DF * max(K - F, 0.0) < p < DF * K

    if (not inRange) and printInfo:
        print
        "The price %1.2f for strike %1.2f and forward %1.2f is NOT arbitrage-free." % (p, K, F)

    return inRange


def impliedVolJaeckelRational(F, K, price, T, DF=1.0, q=1.0, printInfo=False, tolerance=1e-8, itermax=2):
    """ Computation of the Implied Volatility for a Call option by Peter Jaeckel *Lets Be Rational* algorithm
        (2015).

        Inputs:

            @F: the underlyer value
            @K: the strike
            @price: the input price
            @T: the maturity (in years)

            @DF: the discount factor (default to 1.0)
            @q: 1.0 for a Call, -1.0 for a Put (default to 1.0)

            @printInfo: flag used in the isArbitrageFreePrice function
            @tolerance (1e-8): relative tolerance in the iteration procedure
            @itermax (2): maximum number of iterations

        Output:

            The Black Scholes implied volatility, or NaN if the price p is not
                arbitrage-free.

        Side effect:

            impliedVolJaeckelRational.iter contains the effective number of iterations
    """
    x = log(F / K)
    beta = price / (DF * sqrt(F * K))

    if not isArbitrageFreePrice(F, K, price, DF, q, printInfo=printInfo): return float('nan')

    sigma, nbrIter = JaeckelRational(x, beta, q, relTol=tolerance, N=itermax)

    impliedVolJaeckelRational.iter = nbrIter

    return sigma / sqrt(T)


impliedVolJaeckelRational.iter = 0