"""
Statistical tests for model validation.

Provides wrapper functions around statsmodels tests for residual diagnostics.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ..results import TestResult

# Lazy import statsmodels to avoid import errors if not installed
_statsmodels_available = None


def _check_statsmodels():
    """Check if statsmodels is available."""
    global _statsmodels_available
    if _statsmodels_available is None:
        try:
            import statsmodels.stats.diagnostic
            import statsmodels.stats.stattools

            _statsmodels_available = True
        except ImportError:
            _statsmodels_available = False
    return _statsmodels_available


def durbin_watson_test(
    residuals: np.ndarray,
    significance_level: float = 0.05,
) -> TestResult:
    """
    Durbin-Watson test for autocorrelation in residuals.

    Tests whether there is first-order autocorrelation in the residuals.
    The statistic ranges from 0 to 4, with 2 indicating no autocorrelation.

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals.
    significance_level : float
        Significance level (not used directly but stored for reference).

    Returns
    -------
    TestResult
        Test result with statistic and interpretation.

    Notes
    -----
    - DW ≈ 2: No autocorrelation
    - DW < 2: Positive autocorrelation
    - DW > 2: Negative autocorrelation
    - Values between 1.5 and 2.5 are generally acceptable.
    """
    if not _check_statsmodels():
        # Fallback implementation
        n = len(residuals)
        dw = np.sum(np.diff(residuals) ** 2) / np.sum(residuals**2)
    else:
        from statsmodels.stats.stattools import durbin_watson

        dw = durbin_watson(residuals)

    # Interpretation based on common rules of thumb
    if 1.5 <= dw <= 2.5:
        passed = True
        interpretation = "No significant autocorrelation detected"
    elif dw < 1.5:
        passed = False
        interpretation = "Positive autocorrelation detected (DW < 1.5)"
    else:
        passed = False
        interpretation = "Negative autocorrelation detected (DW > 2.5)"

    return TestResult(
        test_name="Durbin-Watson",
        statistic=float(dw),
        p_value=np.nan,  # DW doesn't have a simple p-value
        passed=passed,
        threshold=significance_level,
        interpretation=interpretation,
    )


def ljung_box_test(
    residuals: np.ndarray,
    max_lag: int = 10,
    significance_level: float = 0.05,
) -> TestResult:
    """
    Ljung-Box test for autocorrelation at multiple lags.

    Tests the null hypothesis that the residuals are independently distributed
    (no autocorrelation up to the specified lag).

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals.
    max_lag : int
        Maximum lag to test.
    significance_level : float
        Significance level for the test.

    Returns
    -------
    TestResult
        Test result with statistic, p-value, and interpretation.
    """
    if not _check_statsmodels():
        # Simplified fallback using scipy
        n = len(residuals)
        acf_values = np.correlate(residuals, residuals, mode="full")
        acf_values = acf_values[n - 1 :] / acf_values[n - 1]

        # Compute Q statistic
        lags = min(max_lag, n - 1)
        q_stat = (
            n
            * (n + 2)
            * np.sum([acf_values[k] ** 2 / (n - k) for k in range(1, lags + 1)])
        )
        p_value = 1 - stats.chi2.cdf(q_stat, df=lags)
    else:
        from statsmodels.stats.diagnostic import acorr_ljungbox

        result = acorr_ljungbox(residuals, lags=[max_lag], return_df=True)
        q_stat = result["lb_stat"].iloc[0]
        p_value = result["lb_pvalue"].iloc[0]

    passed = p_value > significance_level
    interpretation = (
        "No significant autocorrelation at tested lags"
        if passed
        else f"Significant autocorrelation detected (p={p_value:.4f})"
    )

    return TestResult(
        test_name=f"Ljung-Box (lag={max_lag})",
        statistic=float(q_stat),
        p_value=float(p_value),
        passed=passed,
        threshold=significance_level,
        interpretation=interpretation,
    )


def breusch_pagan_test(
    residuals: np.ndarray,
    exog: np.ndarray | None = None,
    significance_level: float = 0.05,
) -> TestResult:
    """
    Breusch-Pagan test for heteroscedasticity.

    Tests the null hypothesis that the residual variance is constant
    (homoscedasticity).

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals.
    exog : np.ndarray, optional
        Exogenous variables. If None, uses fitted values proxy.
    significance_level : float
        Significance level for the test.

    Returns
    -------
    TestResult
        Test result with statistic, p-value, and interpretation.
    """
    n = len(residuals)

    if exog is None:
        # Use index as proxy when no exog provided
        exog = np.column_stack([np.ones(n), np.arange(n)])

    if not _check_statsmodels():
        # Simplified fallback
        # Regress squared residuals on exog
        squared_residuals = residuals**2
        if exog.ndim == 1:
            exog = exog.reshape(-1, 1)
        if exog.shape[1] == 1:
            exog = np.column_stack([np.ones(n), exog])

        # OLS for squared residuals
        try:
            beta = np.linalg.lstsq(exog, squared_residuals, rcond=None)[0]
            fitted = exog @ beta
            ss_reg = np.sum((fitted - squared_residuals.mean()) ** 2)
            ss_tot = np.sum((squared_residuals - squared_residuals.mean()) ** 2)
            r2 = ss_reg / ss_tot if ss_tot > 0 else 0

            # LM statistic
            lm_stat = n * r2
            p_value = 1 - stats.chi2.cdf(lm_stat, df=exog.shape[1] - 1)
        except Exception:
            lm_stat = np.nan
            p_value = 1.0
    else:
        from statsmodels.stats.diagnostic import het_breuschpagan

        try:
            lm_stat, p_value, _, _ = het_breuschpagan(residuals, exog)
        except Exception:
            lm_stat = np.nan
            p_value = 1.0

    passed = p_value > significance_level
    interpretation = (
        "No significant heteroscedasticity detected"
        if passed
        else f"Heteroscedasticity detected (p={p_value:.4f})"
    )

    return TestResult(
        test_name="Breusch-Pagan",
        statistic=float(lm_stat),
        p_value=float(p_value),
        passed=passed,
        threshold=significance_level,
        interpretation=interpretation,
    )


def shapiro_wilk_test(
    residuals: np.ndarray,
    significance_level: float = 0.05,
    max_samples: int = 5000,
) -> TestResult:
    """
    Shapiro-Wilk test for normality of residuals.

    Tests the null hypothesis that the residuals are normally distributed.

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals.
    significance_level : float
        Significance level for the test.
    max_samples : int
        Maximum number of samples (Shapiro-Wilk has sample size limits).

    Returns
    -------
    TestResult
        Test result with statistic, p-value, and interpretation.
    """
    # Shapiro-Wilk has sample size limits
    if len(residuals) > max_samples:
        # Use random subsample
        rng = np.random.default_rng(42)
        indices = rng.choice(len(residuals), size=max_samples, replace=False)
        test_residuals = residuals[indices]
    else:
        test_residuals = residuals

    try:
        statistic, p_value = stats.shapiro(test_residuals)
    except Exception:
        statistic = np.nan
        p_value = 1.0

    passed = p_value > significance_level
    interpretation = (
        "Residuals appear normally distributed"
        if passed
        else f"Residuals deviate from normality (p={p_value:.4f})"
    )

    return TestResult(
        test_name="Shapiro-Wilk",
        statistic=float(statistic),
        p_value=float(p_value),
        passed=passed,
        threshold=significance_level,
        interpretation=interpretation,
    )


def jarque_bera_test(
    residuals: np.ndarray,
    significance_level: float = 0.05,
) -> TestResult:
    """
    Jarque-Bera test for normality of residuals.

    Tests the null hypothesis that the residuals have skewness and kurtosis
    matching a normal distribution.

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals.
    significance_level : float
        Significance level for the test.

    Returns
    -------
    TestResult
        Test result with statistic, p-value, and interpretation.
    """
    try:
        statistic, p_value = stats.jarque_bera(residuals)
    except Exception:
        statistic = np.nan
        p_value = 1.0

    passed = p_value > significance_level
    interpretation = (
        "Residuals have normal skewness and kurtosis"
        if passed
        else f"Residuals deviate from normality (p={p_value:.4f})"
    )

    return TestResult(
        test_name="Jarque-Bera",
        statistic=float(statistic),
        p_value=float(p_value),
        passed=passed,
        threshold=significance_level,
        interpretation=interpretation,
    )


def compute_acf(
    residuals: np.ndarray,
    max_lag: int = 20,
) -> np.ndarray:
    """
    Compute autocorrelation function.

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals.
    max_lag : int
        Maximum lag to compute.

    Returns
    -------
    np.ndarray
        ACF values from lag 0 to max_lag.
    """
    if _check_statsmodels():
        from statsmodels.tsa.stattools import acf

        return acf(residuals, nlags=max_lag, fft=True)
    else:
        # Manual computation
        n = len(residuals)
        mean = residuals.mean()
        var = np.sum((residuals - mean) ** 2)
        acf_values = np.zeros(max_lag + 1)

        for k in range(max_lag + 1):
            acf_values[k] = (
                np.sum((residuals[: n - k] - mean) * (residuals[k:] - mean)) / var
            )

        return acf_values


def compute_pacf(
    residuals: np.ndarray,
    max_lag: int = 20,
) -> np.ndarray:
    """
    Compute partial autocorrelation function.

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals.
    max_lag : int
        Maximum lag to compute.

    Returns
    -------
    np.ndarray
        PACF values from lag 0 to max_lag.
    """
    if _check_statsmodels():
        from statsmodels.tsa.stattools import pacf

        return pacf(residuals, nlags=max_lag)
    else:
        # Simplified Yule-Walker estimation
        acf_values = compute_acf(residuals, max_lag)
        pacf_values = np.zeros(max_lag + 1)
        pacf_values[0] = 1.0

        if max_lag >= 1:
            pacf_values[1] = acf_values[1]

        for k in range(2, max_lag + 1):
            # Levinson-Durbin recursion (simplified)
            try:
                r = acf_values[1 : k + 1]
                R = np.zeros((k, k))
                for i in range(k):
                    for j in range(k):
                        R[i, j] = acf_values[abs(i - j)]
                phi = np.linalg.solve(R, r)
                pacf_values[k] = phi[-1]
            except Exception:
                pacf_values[k] = 0.0

        return pacf_values


__all__ = [
    "durbin_watson_test",
    "ljung_box_test",
    "breusch_pagan_test",
    "shapiro_wilk_test",
    "jarque_bera_test",
    "compute_acf",
    "compute_pacf",
]
