"""Phase 3.3 aperture photometry on synthetic SPHEREx products."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from app.core.cache import CacheStore
from app.schemas.photometry import PhotometryConfig, PhotometryRequest
from app.services.image_preparation import prepare_image
from app.services.observation_service import ObservationService
from app.services.photometry import ObservationContext, PhotometryService, measure_prepared
from app.services.source_detection import SourceDetectionService, detect_fits
from tests.conftest import TARGET_DEC, TARGET_RA, make_sky_fits_bytes, seed_pair

SIGMA_ARCSEC = 12.0
SIGMA_PX = SIGMA_ARCSEC / 6.15
PSF_FWHM_PX = SIGMA_PX * 2.3548200450309493
EXPECTED_JY = 1e-2 * 2.0 * math.pi * (SIGMA_ARCSEC / 206264.80624709636) ** 2 * 1e6


def _write(tmp_path, **kwargs):
    path = tmp_path / "phot.fits"
    path.write_bytes(make_sky_fits_bytes(**kwargs))
    return path


def _measure(tmp_path, **kwargs):
    path = _write(tmp_path, **kwargs)
    result = detect_fits(path, observation_key="TEST")
    prepared = prepare_image(path)
    measurements = measure_prepared(
        prepared,
        result.detections,
        PhotometryConfig(),
        context=ObservationContext(
            observation_id="token",
            obs_publisher_did="ivo://irsa.ipac/spherex_qr3?TEST",
            band="SPHEREx-D3",
            observation_time="2026-09-24T06:36:54Z",
            time_mjd=61177.275636,
        ),
    )
    return result, prepared, measurements


def test_integrated_flux_matches_the_rendered_gaussian(tmp_path):
    result, prepared, measurements = _measure(
        tmp_path,
        sources=((0.0, 0.0, 1e-2, SIGMA_ARCSEC),),
        noise=1e-5,
        with_psf=True,
        psf_fwhm_px=PSF_FWHM_PX,
    )

    assert len(result.detections) == 1
    measurement = measurements[0]
    assert measurement.fwhm_source == "psf"
    assert measurement.background_method == "annulus_sigma_clipped"
    assert measurement.aperture_sum_unit == "MJy / sr"
    assert measurement.solid_angle_sr == pytest.approx(prepared.pixel_solid_angle_sr)
    assert measurement.flux_mjy == pytest.approx(EXPECTED_JY / 1e6, rel=0.02)
    assert measurement.flux_jy == pytest.approx(EXPECTED_JY, rel=0.02)
    assert measurement.flux_jy_error is not None and measurement.flux_jy_error > 0
    assert measurement.snr is not None and measurement.snr > 10
    assert measurement.centroid_error_arcsec is not None
    assert measurement.centroid_error_arcsec < 0.2
    assert measurement.uncertainty_method in {"variance_array", "mixed"}
    assert measurement.quality == "PASS"
    assert measurement.flags == []
    assert measurement.aperture_correction_applied is False
    assert measurement.observation_id == "token"
    assert measurement.band == "SPHEREx-D3"


def test_missing_variance_falls_back_to_background_uncertainty(tmp_path):
    _result, _prepared, measurements = _measure(
        tmp_path,
        sources=((0.0, 0.0, 1e-2, SIGMA_ARCSEC),),
        noise=1e-5,
        with_psf=True,
        psf_fwhm_px=PSF_FWHM_PX,
        with_variance=False,
    )

    measurement = measurements[0]
    assert measurement.uncertainty_method == "background_sigma"
    assert measurement.flux_jy_error is not None and measurement.flux_jy_error > 0


def test_surface_brightness_without_solid_angle_is_not_converted(tmp_path):
    _result, _prepared, measurements = _measure(
        tmp_path,
        sources=((0.0, 0.0, 1e-2, SIGMA_ARCSEC),),
        noise=1e-5,
        with_psf=True,
        psf_fwhm_px=PSF_FWHM_PX,
        with_wcs=False,
    )

    measurement = measurements[0]
    assert measurement.solid_angle_sr is None
    assert measurement.flux_mjy is None
    assert measurement.flux_jy is None
    assert "missing_solid_angle" in measurement.flags
    assert measurement.quality == "WARNING"


def test_unsupported_unit_keeps_native_sum_without_guessing_jy(tmp_path):
    _result, _prepared, measurements = _measure(
        tmp_path,
        sources=((0.0, 0.0, 1e-2, SIGMA_ARCSEC),),
        noise=1e-5,
        with_psf=True,
        psf_fwhm_px=PSF_FWHM_PX,
        bunit="Jy",
    )

    measurement = measurements[0]
    assert measurement.aperture_sum is not None
    assert measurement.flux_jy is None
    assert "unit_unsupported" in measurement.flags
    assert measurement.quality == "WARNING"


def test_unusable_annulus_falls_back_to_global_background(tmp_path):
    yy, xx = np.mgrid[0:64, 0:64]
    distance = np.hypot(xx - 31.5, yy - 31.5)
    flags = np.where((distance >= 11.5) & (distance <= 19.0), 1, 0).astype(np.int32)
    _result, _prepared, measurements = _measure(
        tmp_path,
        sources=((0.0, 0.0, 1e-2, SIGMA_ARCSEC),),
        noise=1e-5,
        flags=flags,
        mask_bits={"BADPIX": "unusable"},
    )

    assert measurements
    measurement = measurements[0]
    assert measurement.background_method == "global"
    assert measurement.flux_jy is not None


def test_fully_masked_aperture_fails():
    path = Path(tempfile.mkdtemp()) / "masked.fits"
    path.write_bytes(
        make_sky_fits_bytes(
            sources=((0.0, 0.0, 1e-2, SIGMA_ARCSEC),),
            noise=1e-5,
            with_psf=True,
            psf_fwhm_px=PSF_FWHM_PX,
        )
    )
    result = detect_fits(path, observation_key="TEST")
    prepared = prepare_image(path)
    prepared.valid[:] = False
    measurements = measure_prepared(prepared, result.detections, PhotometryConfig())

    measurement = measurements[0]
    assert measurement.aperture_valid_fraction == 0.0
    assert "low_valid_coverage" in measurement.flags
    assert measurement.flux_jy is None
    assert measurement.quality in {"FAIL", "NOT_VERIFIED"}


def test_service_measures_and_reuses_the_cache(settings, fake_client):
    fits = make_sky_fits_bytes(
        sources=((0.0, 0.0, 1e-2, SIGMA_ARCSEC),),
        noise=1e-5,
        with_psf=True,
        psf_fwhm_px=PSF_FWHM_PX,
    )
    token_a, _token_b = seed_pair(fake_client, fits, fits)
    observations = ObservationService(fake_client, CacheStore(60, 32), settings)
    detection = SourceDetectionService(observations)
    service = PhotometryService(observations, detection)
    request = PhotometryRequest(
        observation_id=token_a,
        ra=TARGET_RA,
        dec=TARGET_DEC,
        size=0.05,
    )

    response = service.measure(request)
    assert response.total_measurements == 1
    assert response.measurements[0].observation_id == token_a
    assert response.measurements[0].obs_publisher_did
    assert response.provenance.fits_sha256 == response.detection.fits_sha256

    calls_before = len(fake_client.session.calls)
    second = service.measure(request)
    assert second is response
    assert len(fake_client.session.calls) == calls_before
