"""Phase 3.2 source detection on synthetic SPHEREx-shaped products."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.cache import CacheStore
from app.schemas.sources import DetectionConfig, DetectionRequest
from app.services.observation_service import ObservationService
from app.services.source_detection import SourceDetectionService, detect_fits
from tests.conftest import TARGET_DEC, TARGET_RA, make_sky_fits_bytes, seed_pair


def _write(tmp_path, **kwargs):
    path = tmp_path / "detect.fits"
    path.write_bytes(make_sky_fits_bytes(**kwargs))
    return path


def test_detects_the_three_rendered_sources(tmp_path):
    path = _write(
        tmp_path,
        noise=1e-4,
        with_psf=True,
        sources=((0.0, 0.0, 1e-2, 12.0), (150.0, 0.0, 6e-3, 10.0), (-150.0, 0.0, 4e-3, 14.0)),
    )
    result = detect_fits(path, observation_key="TEST")

    detections = result.detections
    assert len(detections) == 3
    assert result.quality.overall_status == "PASS"
    assert result.quality.warnings == []

    central = detections[0]
    assert central.detection_snr > 10
    assert abs(central.x - 31.5) < 1.5
    assert abs(central.y - 31.5) < 1.5
    assert central.ra_deg == pytest.approx(TARGET_RA, abs=2e-3)
    assert central.dec_deg == pytest.approx(TARGET_DEC, abs=2e-3)
    assert central.quality == "PASS"
    assert central.flags == []

    for detection in detections:
        assert len(detection.detection_id) == 32
        assert detection.source_area >= 5
        assert detection.detection_snr is not None
        assert detection.centroid_error_px is not None and detection.centroid_error_px > 0
        assert detection.centroid_error_arcsec is not None
        assert detection.centroid_error_arcsec == pytest.approx(
            detection.centroid_error_px * 6.15, rel=1e-4
        )
        assert detection.fwhm_px is not None and detection.fwhm_px > 0
        assert detection.ellipticity is not None
        assert detection.background_level is not None
        assert detection.noise_sigma is not None
        assert detection.peak_value is not None


def test_overlapping_bright_sources_are_flagged_not_merged_silently(tmp_path):
    path = _write(tmp_path, noise=1e-4)
    result = detect_fits(path, observation_key="TEST")

    assert len(result.detections) == 3
    assert "blended_sources_present" in result.quality.warnings
    assert any("blended" in item.flags for item in result.detections)
    assert result.quality.overall_status == "WARNING"


def test_detection_ids_are_deterministic(tmp_path):
    path = _write(tmp_path, noise=1e-4)
    first = detect_fits(path, observation_key="TEST")
    second = detect_fits(path, observation_key="TEST")

    assert [item.detection_id for item in first.detections] == [
        item.detection_id for item in second.detections
    ]


def test_detection_ids_depend_on_the_product_identity(tmp_path):
    path = _write(tmp_path, noise=1e-4)
    first = detect_fits(path, observation_key="PRODUCT_A")
    second = detect_fits(path, observation_key="PRODUCT_B")

    assert {item.detection_id for item in first.detections}.isdisjoint(
        {item.detection_id for item in second.detections}
    )


def test_max_sources_truncates_and_is_reported(tmp_path):
    path = _write(tmp_path, noise=1e-4)
    result = detect_fits(path, DetectionConfig(max_sources=1), observation_key="TEST")

    assert len(result.detections) == 1
    assert result.truncated is True
    assert "detection_limit_applied" in result.quality.warnings
    assert result.quality.overall_status == "WARNING"


def test_no_valid_pixels_fails_without_fabricating_detections(tmp_path):
    path = _write(tmp_path, fill_value=np.nan)
    result = detect_fits(path, observation_key="TEST")

    assert result.detections == []
    assert result.quality.overall_status == "FAIL"
    assert "no_valid_pixels" in result.quality.warnings


def test_flat_image_is_not_verified(tmp_path):
    path = _write(tmp_path, fill_value=0.0, noise=0.0)
    result = detect_fits(path, observation_key="TEST")

    assert result.detections == []
    assert result.quality.overall_status == "NOT_VERIFIED"
    assert "background_or_noise_unavailable" in result.quality.warnings


def test_missing_wcs_keeps_pixel_detections_and_flags_them(tmp_path):
    path = _write(tmp_path, noise=1e-4, with_wcs=False)
    result = detect_fits(path, observation_key="TEST")

    assert result.detections
    assert all(item.ra_deg is None and item.dec_deg is None for item in result.detections)
    assert all("wcs_unavailable" in item.flags for item in result.detections)
    assert "missing_wcs" in result.quality.warnings
    assert result.quality.overall_status == "WARNING"


def test_noise_only_image_produces_no_sources(tmp_path):
    path = _write(
        tmp_path,
        sources=(),
        noise=1e-4,
        background=1e-3,
    )
    result = detect_fits(path, observation_key="TEST")

    assert result.detections == []
    assert "no_sources_detected" in result.quality.warnings


def test_service_uses_cutout_cache_and_returns_provenance(settings, fake_client):
    a_bytes = make_sky_fits_bytes(noise=1e-4)
    token_a, _token_b = seed_pair(fake_client, a_bytes, a_bytes)
    service = SourceDetectionService(
        ObservationService(fake_client, CacheStore(60, 32), settings)
    )
    request = DetectionRequest(
        observation_id=token_a,
        ra=TARGET_RA,
        dec=TARGET_DEC,
        size=0.05,
    )

    response = service.detect(request)
    assert response.source == "cutout"
    assert response.total_detections >= 1
    assert response.provenance.algorithm_version == "phase3-source-detection-v1"
    assert len(response.provenance.fits_sha256) == 64
    assert response.provenance.obs_publisher_did
    assert response.image.pixel_solid_angle_sr is not None

    calls_before = len(fake_client.session.calls)
    second = service.detect(request)
    assert second is response
    assert len(fake_client.session.calls) == calls_before


def test_service_rejects_half_a_coordinate(settings, fake_client):
    from app.core.exceptions import InvalidCoordinateError

    service = SourceDetectionService(
        ObservationService(fake_client, CacheStore(60, 32), settings)
    )
    request = DetectionRequest(observation_id="token", ra=TARGET_RA)

    try:
        service.detect(request)
    except InvalidCoordinateError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("expected InvalidCoordinateError")
