"""Phase 3.5 light-curve construction and end-to-end target orchestration."""

from __future__ import annotations

import math

import pytest

from app.core.cache import CacheStore
from app.schemas.lightcurve import (
    AssociationConfig,
    EpochMeasurements,
    TargetLightCurveRequest,
)
from app.schemas.photometry import FluxMeasurement
from app.services.lightcurve_service import TargetLightCurveService, build_light_curves
from app.services.observation_service import ObservationService
from app.services.source_association import associate_epochs
from tests.conftest import TARGET_DEC, TARGET_RA, make_sky_fits_bytes, seed_pair

SIGMA_ARCSEC = 12.0
PSF_FWHM_PX = (SIGMA_ARCSEC / 6.15) * 2.3548200450309493
EXPECTED_JY = 1e-2 * 2.0 * math.pi * (SIGMA_ARCSEC / 206264.80624709636) ** 2 * 1e6


def _measurement(
    detection_id: str,
    observation_id: str,
    band: str = "A",
    flux: float = 1.0,
    time_mjd: float = 60000.0,
) -> FluxMeasurement:
    return FluxMeasurement(
        detection_id=detection_id,
        observation_id=observation_id,
        band=band,
        time_mjd=time_mjd,
        ra_deg=TARGET_RA,
        dec_deg=TARGET_DEC,
        x=10.0,
        y=10.0,
        aperture_radius_px=6.0,
        aperture_area_px=100.0,
        aperture_valid_fraction=1.0,
        psf_fwhm_px=2.0,
        fwhm_source="psf",
        flux_jy=flux,
        flux_jy_error=0.1,
        snr=10.0,
        quality="PASS",
    )


def _epoch(key: str, observation_id: str, time_mjd: float, measurements, band="A"):
    return EpochMeasurements(
        epoch_key=key,
        observation_id=observation_id,
        band=band,
        time_utc=f"MJD {time_mjd}",
        time_mjd=time_mjd,
        measurements=measurements,
    )


def test_light_curve_keeps_band_and_counts_non_detections():
    epochs = [
        _epoch("e1", "obs-a", 60000.0, [_measurement("det-a", "obs-a", flux=1.0)]),
        _epoch("e2", "obs-b", 60010.0, [_measurement("det-b", "obs-b", flux=1.2)]),
        _epoch("e3", "obs-c", 60020.0, []),
    ]
    sources, _summary = associate_epochs(epochs, AssociationConfig())
    curves = build_light_curves(sources, epochs, AssociationConfig())

    assert len(curves) == 1
    curve = curves[0]
    assert curve.epoch_count == 2
    assert len(curve.series) == 1
    series = curve.series[0]
    assert series.band == "A"
    assert len(series.points) == 2
    assert series.expected_epochs == 3
    assert series.non_detections == 1
    assert [point.flux_jy for point in series.points] == [1.0, 1.2]
    assert series.metrics.status == "computed"
    assert series.metrics.n_points == 2


def test_bands_are_never_concatenated():
    epochs = [
        _epoch("e1", "obs-a", 60000.0, [_measurement("det-a", "obs-a", band="A")], band="A"),
        _epoch("e1", "obs-a-b", 60000.0, [_measurement("det-ab", "obs-a-b", band="B")], band="B"),
        _epoch("e2", "obs-b", 60010.0, [_measurement("det-b", "obs-b", band="A")], band="A"),
        _epoch("e2", "obs-b-b", 60010.0, [_measurement("det-bb", "obs-b-b", band="B")], band="B"),
    ]
    sources, _summary = associate_epochs(epochs, AssociationConfig())
    curves = build_light_curves(sources, epochs, AssociationConfig())

    assert len(sources) == 1
    assert len(curves[0].series) == 2
    assert {series.band for series in curves[0].series} == {"A", "B"}
    assert all(len(series.points) == 2 for series in curves[0].series)
    assert all(series.expected_epochs == 2 for series in curves[0].series)


def test_target_service_builds_a_two_epoch_light_curve(settings, fake_client):
    fits = make_sky_fits_bytes(
        sources=((0.0, 0.0, 1e-2, SIGMA_ARCSEC),),
        noise=1e-5,
        with_psf=True,
        psf_fwhm_px=PSF_FWHM_PX,
    )
    seed_pair(fake_client, fits, fits)
    for row in fake_client.rows:
        row["t_max"] = row["t_min"] + 0.01

    observations = ObservationService(fake_client, CacheStore(60, 32), settings)
    service = TargetLightCurveService(observations)
    request = TargetLightCurveRequest(
        ra=TARGET_RA,
        dec=TARGET_DEC,
        radius_deg=0.05,
        cutout_size_deg=0.05,
        max_observations=5,
    )

    response = service.build(request)

    assert response.epochs_processed == 2
    assert response.products_processed == 2
    assert response.failures == []
    assert response.association.source_count == 1
    assert len(response.sources) == 1
    source = response.sources[0]
    assert source.epoch_count == 2
    assert source.bands == ["SPHEREx-D3"]
    assert len(source.series) == 1
    series = source.series[0]
    assert len(series.points) == 2
    assert series.non_detections == 0
    assert series.metrics.status == "computed"
    for point in series.points:
        assert point.unit == "Jy"
        assert point.flux_jy == pytest.approx(EXPECTED_JY, rel=0.05)
        assert point.time_utc is not None
        assert point.time_mjd is not None

    calls_before = len(fake_client.session.calls)
    second = service.build(request)
    assert second is response
    assert len(fake_client.session.calls) == calls_before
