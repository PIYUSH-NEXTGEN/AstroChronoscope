"""Phase 3.5 variability-metric tests on synthetic light curves."""

from __future__ import annotations

import math

import pytest

from app.schemas.lightcurve import LightCurvePoint
from app.services.variability import compute_variability


def _point(time: float, flux: float, error: float | None = 0.1) -> LightCurvePoint:
    return LightCurvePoint(
        time_mjd=time,
        flux_jy=flux,
        flux_jy_error=error,
        observation_id=f"obs-{time}",
        detection_id=f"det-{time}",
        quality="PASS",
    )


def test_constant_series_is_consistent_with_constant_flux():
    points = [_point(60000.0 + index, 1.0) for index in range(5)]
    metrics = compute_variability(points)

    assert metrics.status == "computed"
    assert metrics.n_points == 5
    assert metrics.mean_flux_jy == pytest.approx(1.0)
    assert metrics.sample_std_jy == pytest.approx(0.0)
    assert metrics.chi2 == pytest.approx(0.0)
    assert metrics.reduced_chi2 == pytest.approx(0.0)
    assert metrics.chi2_pvalue == pytest.approx(1.0)
    assert metrics.fractional_variability is None
    assert "fvar_undefined_or_consistent_with_zero" in metrics.flags
    assert metrics.temporal_baseline_days == pytest.approx(4.0)
    assert metrics.max_gap_days == pytest.approx(1.0)


def test_variable_series_has_positive_fractional_variability():
    fluxes = [1.0, 2.0, 1.5, 3.0, 0.5]
    points = [_point(60000.0 + index, flux) for index, flux in enumerate(fluxes)]
    metrics = compute_variability(points)

    assert metrics.status == "computed"
    assert metrics.sample_std_jy is not None and metrics.sample_std_jy > 0
    assert metrics.fractional_variability is not None
    assert metrics.fractional_variability > 0
    assert metrics.chi2 is not None and metrics.chi2 > 10
    assert metrics.chi2_pvalue is not None and metrics.chi2_pvalue < 0.05
    assert metrics.peak_to_peak_jy == pytest.approx(2.5)
    assert metrics.min_flux_jy == pytest.approx(0.5)
    assert metrics.max_flux_jy == pytest.approx(3.0)


def test_missing_uncertainties_are_reported_not_invented():
    points = [_point(60000.0, 1.0, error=None), _point(60001.0, 1.2, error=None)]
    metrics = compute_variability(points)

    assert metrics.status == "undefined_uncertainties"
    assert metrics.weighted_mean_flux_jy is None
    assert metrics.chi2 is None
    assert metrics.reduced_chi2 is None
    assert "missing_uncertainties" in metrics.flags
    assert metrics.sample_std_jy is not None


def test_single_point_is_insufficient_for_variability():
    metrics = compute_variability([_point(60000.0, 1.0)])

    assert metrics.status == "insufficient_data"
    assert metrics.n_points == 1
    assert metrics.sample_std_jy is None
    assert metrics.chi2 is None
    assert "few_epochs" in metrics.flags


def test_zero_flux_does_not_divide_by_zero():
    metrics = compute_variability([_point(60000.0, 0.0), _point(60001.0, 0.0)])
    assert metrics.status == "computed"
    assert metrics.fractional_variability is None
    assert math.isfinite(metrics.mean_flux_jy or 0.0)


def test_empty_series_is_insufficient_data():
    metrics = compute_variability([])
    assert metrics.status == "insufficient_data"
    assert metrics.n_points == 0
    assert "no_valid_points" in metrics.flags
