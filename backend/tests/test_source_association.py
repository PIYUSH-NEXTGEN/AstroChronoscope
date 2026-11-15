"""Phase 3.4 angular association tests."""

from __future__ import annotations

import math

import pytest

from app.schemas.lightcurve import AssociationConfig, EpochMeasurements
from app.schemas.photometry import FluxMeasurement
from app.services.source_association import associate_epochs
from tests.conftest import TARGET_DEC, TARGET_RA


def _measurement(
    detection_id: str,
    observation_id: str,
    *,
    ra: float | None = TARGET_RA,
    dec: float | None = TARGET_DEC,
    band: str | None = "A",
    snr: float | None = 10.0,
    centroid_error_arcsec: float | None = 0.2,
) -> FluxMeasurement:
    return FluxMeasurement(
        detection_id=detection_id,
        observation_id=observation_id,
        band=band,
        time_mjd=60000.0,
        ra_deg=ra,
        dec_deg=dec,
        x=10.0,
        y=10.0,
        centroid_error_arcsec=centroid_error_arcsec,
        aperture_radius_px=6.0,
        aperture_area_px=100.0,
        aperture_valid_fraction=1.0,
        psf_fwhm_px=2.0,
        fwhm_source="psf",
        flux_jy=1.0,
        flux_jy_error=0.1,
        snr=snr,
        quality="PASS",
    )


def _offset(ra: float, dec: float, dx_arcsec: float, dy_arcsec: float) -> tuple[float, float]:
    return (
        ra + dx_arcsec / 3600.0 / math.cos(math.radians(dec)),
        dec + dy_arcsec / 3600.0,
    )


def _epoch(key: str, observation_id: str, time_mjd: float, measurements, band="A"):
    return EpochMeasurements(
        epoch_key=key,
        observation_id=observation_id,
        band=band,
        time_mjd=time_mjd,
        measurements=measurements,
    )


def test_same_source_across_epochs_gets_one_stable_id():
    first = _measurement("det-a", "obs-a")
    second = _measurement("det-b", "obs-b")
    sources, summary = associate_epochs(
        [
            _epoch("e1", "obs-a", 60000.0, [first]),
            _epoch("e2", "obs-b", 60010.0, [second]),
        ],
        AssociationConfig(),
    )

    assert summary.source_count == 1
    assert len(sources) == 1
    source = sources[0]
    assert source.epoch_count == 2
    assert source.detection_count == 2
    assert source.bands == ["A"]
    assert source.position_scatter_arcsec is not None and source.position_scatter_arcsec < 1e-6
    separations = [item.separation_arcsec for item in source.detections]
    assert separations[1] is not None and separations[1] < 1e-6


def test_small_offset_is_matched_and_recorded():
    ra, dec = _offset(TARGET_RA, TARGET_DEC, 3.0, 0.0)
    sources, _summary = associate_epochs(
        [
            _epoch("e1", "obs-a", 60000.0, [_measurement("det-a", "obs-a")]),
            _epoch("e2", "obs-b", 60010.0, [_measurement("det-b", "obs-b", ra=ra, dec=dec)]),
        ],
        AssociationConfig(),
    )

    source = sources[0]
    assert source.detection_count == 2
    matched = source.detections[1]
    assert matched.separation_arcsec is not None
    assert abs(matched.separation_arcsec - 3.0) < 0.05
    assert [candidate.selected for candidate in matched.candidates] == [True]


def test_large_offset_creates_two_single_epoch_sources():
    ra, dec = _offset(TARGET_RA, TARGET_DEC, 30.0, 0.0)
    sources, summary = associate_epochs(
        [
            _epoch("e1", "obs-a", 60000.0, [_measurement("det-a", "obs-a")]),
            _epoch("e2", "obs-b", 60010.0, [_measurement("det-b", "obs-b", ra=ra, dec=dec)]),
        ],
        AssociationConfig(),
    )

    assert summary.source_count == 2
    assert summary.multi_epoch_sources == 0
    assert summary.below_min_epoch_sources == 2
    assert sorted(summary.below_min_epoch_source_ids) == sorted(
        source.source_id for source in sources
    )
    assert all("single_epoch" in source.flags for source in sources)


def test_positional_uncertainty_can_expand_the_match_radius_within_a_cap():
    ra, dec = _offset(TARGET_RA, TARGET_DEC, 7.0, 0.0)
    sources, summary = associate_epochs(
        [
            _epoch(
                "e1",
                "obs-a",
                60000.0,
                [_measurement("det-a", "obs-a", centroid_error_arcsec=1.0)],
            ),
            _epoch(
                "e2",
                "obs-b",
                60010.0,
                [
                    _measurement(
                        "det-b",
                        "obs-b",
                        ra=ra,
                        dec=dec,
                        centroid_error_arcsec=1.0,
                    )
                ],
            ),
        ],
        AssociationConfig(match_radius_arcsec=6.0, max_uncertainty_boost_arcsec=3.0),
    )

    assert summary.source_count == 1
    source = sources[0]
    assert source.detection_count == 2
    matched = source.detections[1]
    assert matched.position_sigma_arcsec == pytest.approx(math.sqrt(2.0))
    assert matched.effective_match_radius_arcsec == pytest.approx(6.0 + 3.0)
    assert "position_uncertainty_incomplete" not in source.flags


def test_missing_centroid_uncertainty_is_flagged_not_invented():
    sources, _summary = associate_epochs(
        [
            _epoch(
                "e1",
                "obs-a",
                60000.0,
                [_measurement("det-a", "obs-a", centroid_error_arcsec=None)],
            )
        ],
        AssociationConfig(),
    )

    assert "position_uncertainty_incomplete" in sources[0].flags


def test_duplicate_detection_in_one_epoch_is_excluded_with_reason():
    duplicate = _measurement("det-a", "obs-a")
    sources, summary = associate_epochs(
        [_epoch("e1", "obs-a", 60000.0, [duplicate, duplicate])],
        AssociationConfig(),
    )

    assert summary.duplicates_dropped == 1
    assert summary.excluded[0].reason == "duplicate_same_epoch"
    assert len(sources) == 1


def test_close_sources_in_one_epoch_are_kept_and_marked_ambiguous():
    ra_b, dec_b = _offset(TARGET_RA, TARGET_DEC, 4.0, 0.0)
    sources, summary = associate_epochs(
        [
            _epoch(
                "e1",
                "obs-a",
                60000.0,
                [
                    _measurement("det-a", "obs-a"),
                    _measurement("det-b", "obs-a", ra=ra_b, dec=dec_b),
                ],
            )
        ],
        AssociationConfig(),
    )

    assert summary.source_count == 2
    assert summary.ambiguous_matches == 1
    assert any("same_epoch_close_pair" in source.flags for source in sources)


def test_ambiguous_cross_epoch_match_is_flagged_not_forced():
    ra_b, dec_b = _offset(TARGET_RA, TARGET_DEC, 4.0, 0.0)
    ra_mid, dec_mid = _offset(TARGET_RA, TARGET_DEC, 2.0, 0.0)
    sources, summary = associate_epochs(
        [
            _epoch(
                "e1",
                "obs-a",
                60000.0,
                [
                    _measurement("det-a", "obs-a"),
                    _measurement("det-b", "obs-a", ra=ra_b, dec=dec_b),
                ],
            ),
            _epoch(
                "e2",
                "obs-b",
                60010.0,
                [_measurement("det-c", "obs-b", ra=ra_mid, dec=dec_mid)],
            ),
        ],
        AssociationConfig(),
    )

    assert summary.ambiguous_matches >= 1
    assert any("ambiguous_match" in source.flags for source in sources)


def test_detection_without_sky_position_is_preserved_as_excluded():
    sources, summary = associate_epochs(
        [
            _epoch(
                "e1",
                "obs-a",
                60000.0,
                [_measurement("det-a", "obs-a", ra=None, dec=None)],
            )
        ],
        AssociationConfig(),
    )

    assert sources == []
    assert summary.missing_position_dropped == 1
    assert summary.excluded[0].reason == "missing_position"
