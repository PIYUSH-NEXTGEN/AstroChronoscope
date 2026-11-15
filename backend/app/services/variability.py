"""Transparent variability metrics for one band.

These are descriptive statistics and evidence, not a classification. A large
chi-square or fractional variability does not by itself identify a real variable
or transient; it must be reviewed against quality flags and the observing
baseline, exactly as the project's scientific guidance requires.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from app.schemas.lightcurve import VARIABILITY_ALGORITHM_VERSION, LightCurvePoint, VariabilityMetrics


def compute_variability(points: list[LightCurvePoint]) -> VariabilityMetrics:
    usable = [point for point in points if math.isfinite(point.flux_jy)]
    n = len(usable)
    if n == 0:
        return VariabilityMetrics(
            algorithm_version=VARIABILITY_ALGORITHM_VERSION,
            status="insufficient_data",
            n_points=0,
            flags=["no_valid_points"],
        )

    flux = np.asarray([point.flux_jy for point in usable], dtype=np.float64)
    errors = np.asarray(
        [point.flux_jy_error if point.flux_jy_error is not None else np.nan for point in usable],
        dtype=np.float64,
    )
    mean = float(np.mean(flux))
    median = float(np.median(flux))
    sample_std = float(np.std(flux, ddof=1)) if n > 1 else None
    mad = float(np.median(np.abs(flux - median)))
    robust_sigma = 1.4826 * mad
    low, high = (float(value) for value in np.percentile(flux, [16.0, 84.0]))
    flags: list[str] = []
    if n < 3:
        flags.append("few_epochs")

    errors_ok = bool(np.all(np.isfinite(errors)) and np.all(errors > 0))
    weighted_mean = None
    chi2 = None
    dof = None
    reduced = None
    pvalue = None
    fractional = None

    if errors_ok:
        weights = 1.0 / errors ** 2
        weighted_mean = float(np.sum(flux * weights) / weights.sum())
        if n > 1:
            chi2 = float(np.sum((flux - weighted_mean) ** 2 / errors ** 2))
            dof = n - 1
            reduced = chi2 / dof
            pvalue = float(stats.chi2.sf(chi2, dof))
        variance = float(np.var(flux, ddof=1)) if n > 1 else 0.0
        mean_error_squared = float(np.mean(errors ** 2))
        radicand = variance - mean_error_squared
        if n > 1 and mean != 0 and radicand > 0:
            fractional = math.sqrt(radicand) / abs(mean)
        else:
            flags.append("fvar_undefined_or_consistent_with_zero")
    else:
        flags.append("missing_uncertainties")

    times = sorted(
        point.time_mjd for point in usable if point.time_mjd is not None
    )
    baseline = float(times[-1] - times[0]) if len(times) >= 2 else None
    gaps = np.diff(times) if len(times) >= 2 else np.asarray([])
    max_gap = float(np.max(gaps)) if gaps.size else None

    status = "computed" if errors_ok and n >= 2 else (
        "undefined_uncertainties" if n >= 2 else "insufficient_data"
    )
    return VariabilityMetrics(
        algorithm_version=VARIABILITY_ALGORITHM_VERSION,
        status=status,
        n_points=n,
        mean_flux_jy=_round(mean),
        median_flux_jy=_round(median),
        weighted_mean_flux_jy=_round(weighted_mean),
        sample_std_jy=_round(sample_std),
        mad_jy=_round(mad),
        robust_sigma_jy=_round(robust_sigma),
        fractional_variability=_round(fractional),
        chi2=_round(chi2),
        dof=dof,
        reduced_chi2=_round(reduced),
        chi2_pvalue=_round(pvalue),
        peak_to_peak_jy=_round(float(np.max(flux) - np.min(flux))),
        robust_range_jy=_round(high - low),
        min_flux_jy=_round(float(np.min(flux))),
        max_flux_jy=_round(float(np.max(flux))),
        temporal_baseline_days=_round(baseline),
        max_gap_days=_round(max_gap),
        flags=sorted(set(flags)),
    )


def _round(value: float | None) -> float | None:
    return round(value, 12) if value is not None and math.isfinite(value) else None


__all__ = ["compute_variability"]
