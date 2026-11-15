"""Phase 3.1 image preparation: validity, variance, background and geometry."""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.services.image_preparation import estimate_background, prepare_image
from tests.conftest import make_sky_fits_bytes


def _write(tmp_path, **kwargs):
    path = tmp_path / "prep.fits"
    path.write_bytes(make_sky_fits_bytes(**kwargs))
    return path


def test_background_and_noise_recover_the_rendered_values(tmp_path):
    path = _write(tmp_path, noise=2e-4, background=1e-3)
    prepared = prepare_image(path)

    assert prepared.bunit == "MJy / sr"
    assert prepared.background.reliable is True
    assert prepared.background.level == pytest.approx(1e-3, rel=0.1)
    assert prepared.background.sigma == pytest.approx(2e-4, rel=0.25)
    assert prepared.noise == pytest.approx(2e-4, rel=0.25)
    assert prepared.background.mesh_variation is not None


def test_pixel_solid_angle_is_the_projection_plane_constant(tmp_path):
    path = _write(tmp_path)
    prepared = prepare_image(path)

    expected = (6.15 / 3600.0) ** 2 * (math.pi / 180.0) ** 2
    assert prepared.pixel_scale_arcsec == pytest.approx(6.15, rel=1e-6)
    assert prepared.pixel_solid_angle_sr == pytest.approx(expected, rel=1e-4)
    summary = prepared.summary()
    assert summary.pixel_solid_angle_model == "projection_plane_constant"


def test_invalid_pixels_are_excluded_not_zeroed(tmp_path):
    path = _write(tmp_path, nan_border=True)
    prepared = prepare_image(path)

    assert prepared.nan_pixels == 64
    assert prepared.inf_pixels == 1
    assert int(prepared.valid[0, :].sum()) == 0
    assert prepared.valid[1, 0] == np.bool_(False)
    assert prepared.valid_fraction < 1.0
    assert prepared.summary().invalid_pixels == int((~prepared.valid).sum())
    assert np.isnan(prepared.data[0, 0])


def test_flags_mark_bad_pixels_but_keep_informational_bits(tmp_path):
    flags = np.zeros((64, 64), dtype=np.int32)
    flags[10:12, 10:12] = 1
    path = _write(
        tmp_path,
        flags=flags,
        mask_bits={"BADPIX": "unusable pixel"},
    )
    prepared = prepare_image(path)

    assert prepared.flag_masks == {"BADPIX": 1}
    assert prepared.masked_pixels == 4
    assert not prepared.valid[10, 10]
    assert prepared.valid[20, 20]
    assert prepared.summary().mask_bits == {"BADPIX": 1}


def test_missing_variance_is_reported_as_missing(tmp_path):
    path = _write(tmp_path, with_variance=False)
    prepared = prepare_image(path)

    assert prepared.variance is None
    assert prepared.variance_valid_fraction is None
    assert prepared.summary().variance_available is False


def test_variance_is_masked_with_the_image(tmp_path):
    variance = np.full((64, 64), 4e-8, dtype=np.float64)
    variance[20, 20] = np.nan
    path = _write(tmp_path, nan_border=True, variance=variance)
    prepared = prepare_image(path)

    assert prepared.variance is not None
    assert np.isnan(prepared.variance[0, 0])
    assert np.isnan(prepared.variance[20, 20])
    assert prepared.variance_valid_fraction is not None
    assert prepared.variance_valid_fraction < 1.0


def test_missing_wcs_keeps_pixel_geometry_but_no_sky_geometry(tmp_path):
    path = _write(tmp_path, with_wcs=False)
    prepared = prepare_image(path)

    assert prepared.wcs is None
    assert prepared.pixel_scale_arcsec is None
    assert prepared.pixel_solid_angle_sr is None
    assert prepared.summary().pixel_solid_angle_model == "unavailable"
    assert prepared.background.level is not None


def test_too_few_valid_pixels_makes_background_unreliable(tmp_path):
    path = _write(tmp_path, nan_border=True, fill_value=None)
    prepared = prepare_image(path)
    # Re-run on a hand-made nearly empty array to exercise the guard directly.
    data = np.full((8, 8), np.nan)
    data[0, 0] = 1.0
    valid = np.isfinite(data)
    estimate = estimate_background(data, valid)

    assert estimate.method == "none"
    assert estimate.reliable is False
    assert estimate.level is None
    assert prepared.valid_fraction > 0


def test_flat_image_has_no_reliable_noise(tmp_path):
    path = _write(tmp_path, fill_value=0.0, noise=0.0)
    prepared = prepare_image(path)

    assert prepared.noise is None
    assert prepared.background.reliable is False
