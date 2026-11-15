"""Opt-in live Phase 3 end-to-end check against real SPHEREx products.

Run with ``RUN_LIVE=1``. The test asserts invariants only; archive counts,
available bands and detections change with the mission data release.
"""

from __future__ import annotations

import os

import pytest

from app.core.cache import CacheStore
from app.core.config import Settings
from app.schemas.lightcurve import TargetLightCurveRequest
from app.services.irsa_client import IrsaClient
from app.services.lightcurve_service import TargetLightCurveService
from app.services.observation_service import ObservationService

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RUN_LIVE") != "1", reason="set RUN_LIVE=1 to query IRSA"),
]

RA, DEC = 156.09328159, -41.64466331
COLLECTION = "spherex_qr3"


@pytest.fixture
def service(tmp_path) -> TargetLightCurveService:
    settings = Settings(
        storage_dir=tmp_path / "storage",
        default_collection=COLLECTION,
        request_timeout=240.0,
        download_retries=4,
    )
    settings.ensure_dirs()
    observations = ObservationService(IrsaClient(settings), CacheStore(60, 512), settings)
    return TargetLightCurveService(observations)


def test_live_lightcurve_and_source_catalogue(service) -> None:
    request = TargetLightCurveRequest(
        ra=RA,
        dec=DEC,
        radius_deg=0.01,
        cutout_size_deg=0.05,
        bands=["SPHEREx-D3"],
        max_observations=2,
    )
    response = service.build(request)

    assert response.epochs_considered > 0
    assert response.epochs_processed > 0
    assert response.epochs_processed <= request.max_observations
    assert response.products_processed >= response.epochs_processed
    assert response.failures == [] or all(failure.code for failure in response.failures)
    assert response.association.source_count == len(response.association.below_min_epoch_source_ids) + len(response.sources)

    for source in response.sources:
        assert source.epoch_count >= request.association.min_epochs
        if not source.series:
            continue
        for series in source.series:
            assert series.band == "SPHEREx-D3"
            for point in series.points:
                assert point.unit == "Jy"
                assert point.flux_jy is not None
                assert point.flux_jy_error is None or point.flux_jy_error >= 0
                assert point.quality in {"PASS", "WARNING", "FAIL", "NOT_VERIFIED"}

        stored = service.catalog.source(source.source_id)
        assert stored.source.source_id == source.source_id
        stored_curve = service.catalog.lightcurve(source.source_id)
        assert len(stored_curve.source.series) == len(source.series)
        stored_variability = service.catalog.variability(source.source_id)
        assert stored_variability.source.source_id == source.source_id
