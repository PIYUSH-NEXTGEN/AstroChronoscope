"""Phase 3 light-curve construction and the target-level orchestration service.

The orchestration follows the project contracts: archive discovery happens
through the existing SIA2/timeline path, one product is chosen per epoch/band,
detections are photometered on their native grids, detections are associated on
the sphere, and each band becomes its own light curve. The service never mixes
bands, never turns a non-detection into zero flux and never labels a source as a
confirmed variable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from datetime import datetime, timezone
from typing import Any

from app.core.cache import CacheStore
from app.core.config import Settings
from app.core.exceptions import PlanetXError
from app.core.validation import validate_cutout_size
from app.core.versions import software_versions
from app.schemas.common import Target
from app.schemas.lightcurve import (
    AssociatedSource,
    AssociationConfig,
    EpochFailure,
    EpochMeasurements,
    LightCurvePoint,
    LightCurveProvenance,
    LightCurveQuery,
    LightCurveSeries,
    SourceLightCurve,
    TargetLightCurveRequest,
    TargetLightCurveResponse,
)
from app.schemas.photometry import PhotometryConfig
from app.schemas.sources import DetectionRequest
from app.schemas.timeline import TimelineEpoch, TimelineProduct
from app.services.observation_service import ObservationService
from app.services.photometry import PhotometryService
from app.services.source_association import associate_epochs
from app.services.source_catalog import SourceCatalogService
from app.services.source_detection import SourceDetectionService
from app.services.timeline_service import TimelineService
from app.services.variability import compute_variability

log = logging.getLogger(__name__)

_LIGHTCURVE_VERSION = "phase3-lightcurve-v1"


def build_light_curves(
    sources: list[AssociatedSource],
    epochs: list[EpochMeasurements],
    config: AssociationConfig,
) -> list[SourceLightCurve]:
    """Turn associated detections into per-band light curves with metrics."""
    epoch_time = {
        epoch.epoch_key: (epoch.time_utc, epoch.time_mjd)
        for epoch in epochs
    }
    epoch_bands = {
        (epoch.epoch_key, epoch.observation_id): epoch.band
        for epoch in epochs
    }
    expected: dict[str | None, set[str]] = {}
    for epoch in epochs:
        expected.setdefault(epoch.band, set()).add(epoch.epoch_key)

    curves: list[SourceLightCurve] = []
    for source in sources:
        if source.epoch_count < config.min_epochs:
            continue
        by_band: dict[str | None, list[LightCurvePoint]] = {}
        for matched in source.detections:
            measurement = matched.measurement
            if measurement.flux_jy is None:
                continue
            band = epoch_bands.get(
                (matched.epoch_key, measurement.observation_id), measurement.band
            )
            time_utc, time_mjd = epoch_time.get(
                matched.epoch_key, (measurement.observation_time, measurement.time_mjd)
            )
            by_band.setdefault(band, []).append(
                LightCurvePoint(
                    time_utc=time_utc,
                    time_mjd=time_mjd,
                    flux_jy=measurement.flux_jy,
                    flux_jy_error=measurement.flux_jy_error,
                    band=band,
                    observation_id=measurement.observation_id,
                    obs_publisher_did=measurement.obs_publisher_did,
                    detection_id=measurement.detection_id,
                    snr=measurement.snr,
                    quality=measurement.quality,
                    flags=list(measurement.flags),
                )
            )

        series: list[LightCurveSeries] = []
        for band, points in sorted(by_band.items(), key=lambda item: (item[0] is None, item[0] or "")):
            points.sort(
                key=lambda item: (
                    item.time_mjd if item.time_mjd is not None else math.inf,
                    item.detection_id,
                )
            )
            expect = len(expected.get(band, set()))
            series.append(
                LightCurveSeries(
                    band=band,
                    points=points,
                    expected_epochs=expect,
                    non_detections=max(0, expect - len(points)),
                    metrics=compute_variability(points),
                )
            )

        flags = list(source.flags)
        if not series:
            flags.append("no_usable_flux")
        curves.append(
            SourceLightCurve(
                source_id=source.source_id,
                reference_ra_deg=source.reference_ra_deg,
                reference_dec_deg=source.reference_dec_deg,
                epoch_count=source.epoch_count,
                bands=source.bands,
                position_scatter_arcsec=source.position_scatter_arcsec,
                series=series,
                flags=sorted(set(flags)),
            )
        )
    return curves


class TargetLightCurveService:
    def __init__(
        self,
        observations: ObservationService,
        detection: SourceDetectionService | None = None,
        photometry: PhotometryService | None = None,
        catalog: SourceCatalogService | None = None,
    ) -> None:
        self.observations = observations
        self.settings: Settings = observations.settings
        self.cache: CacheStore = observations.cache
        self.timeline = TimelineService(observations)
        self.detection = detection or SourceDetectionService(observations)
        self.photometry = photometry or PhotometryService(observations, self.detection)
        self.catalog = catalog or SourceCatalogService(self.settings, self.cache)

    def build(self, request: TargetLightCurveRequest) -> TargetLightCurveResponse:
        cutout_size = validate_cutout_size(
            request.cutout_size_deg, self.settings.max_cutout_size_deg
        )
        photometry_config = request.photometry or self.photometry.config_from_settings()
        key = _cache_key(request, cutout_size, photometry_config)
        cached = self.cache.lightcurves.get(key)
        if cached is not None:
            return cached

        timeline = self.timeline.build(
            request.ra,
            request.dec,
            radius=request.radius_deg,
            collection=request.collection,
            bands=request.bands,
            start_date=request.start_date,
            end_date=request.end_date,
        )
        selected = _select_products(timeline.epochs, request.bands, request.max_observations)

        epochs: list[EpochMeasurements] = []
        failures: list[EpochFailure] = []
        products_processed = 0
        for epoch, product in selected:
            assert product.observation_id is not None
            try:
                detection = self.detection.detect(
                    DetectionRequest(
                        observation_id=product.observation_id,
                        ra=request.ra,
                        dec=request.dec,
                        size=cutout_size,
                        config=request.detection,
                    )
                )
                measured = self.photometry.measure_detection(detection, photometry_config)
            except PlanetXError as exc:
                failures.append(
                    EpochFailure(
                        epoch_key=epoch.epoch_key,
                        observation_id=product.observation_id,
                        band=product.band,
                        code=exc.code,
                        message=str(exc),
                    )
                )
                continue
            products_processed += 1
            epochs.append(
                EpochMeasurements(
                    epoch_key=epoch.epoch_key,
                    observation_id=product.observation_id,
                    obs_publisher_did=product.obs_publisher_did,
                    band=product.band,
                    time_utc=epoch.observation_time,
                    time_mjd=epoch.midpoint_mjd,
                    measurements=measured.measurements,
                    quality=measured.quality,
                    warnings=measured.warnings,
                )
            )

        associated, summary = associate_epochs(epochs, request.association)
        sources = build_light_curves(associated, epochs, request.association)
        warnings = set()
        if summary.ambiguous_matches:
            warnings.add("ambiguous_associations_present")
        if summary.duplicates_dropped:
            warnings.add("duplicate_epoch_detections_dropped")
        if summary.missing_position_dropped:
            warnings.add("detections_without_position_dropped")
        if failures:
            warnings.add("some_observations_failed")
        if summary.below_min_epoch_sources:
            warnings.add("sources_below_minimum_epochs_not_in_lightcurves")
        if summary.sources_truncated:
            warnings.add("association_source_limit_reached")
        if not sources:
            warnings.add("no_multi_epoch_sources")

        response = TargetLightCurveResponse(
            target=Target(ra=request.ra, dec=request.dec),
            query=LightCurveQuery(
                radius_deg=request.radius_deg,
                cutout_size_deg=cutout_size,
                collection=request.collection,
                bands=request.bands,
                start_date=request.start_date,
                end_date=request.end_date,
            ),
            epochs_considered=len(timeline.epochs),
            epochs_processed=len({epoch.epoch_key for epoch in epochs}),
            products_processed=products_processed,
            sources=sources,
            failures=failures,
            association=summary,
            warnings=sorted(warnings),
            provenance=LightCurveProvenance(
                created_at=datetime.now(timezone.utc).isoformat(),
                software=software_versions(),
            ),
        )
        self.catalog.store(response, associated, request.association)
        self.cache.lightcurves.set(key, response)
        log.info(
            "lightcurves_built",
            extra={
                "sources": len(sources),
                "epochs": response.epochs_processed,
                "failures": len(failures),
            },
        )
        return response


def _select_products(
    epochs: list[TimelineEpoch], bands: list[str] | None, limit: int
) -> list[tuple[TimelineEpoch, TimelineProduct]]:
    """Choose one ready product per band for at most ``limit`` epochs.

    ``max_observations`` is an epoch limit, not a product-row limit: all requested
    bands for a selected epoch are kept together so a light curve is not truncated
    mid-epoch.
    """
    wanted = {band.casefold() for band in bands} if bands else None
    selected: list[tuple[TimelineEpoch, TimelineProduct]] = []
    selected_epochs = 0
    for epoch in epochs:
        if selected_epochs >= limit:
            break
        if not epoch.timeline_eligible or epoch.target_covered is False:
            continue
        if wanted is None:
            target_bands: list[str | None] = list(epoch.bands) or [None]
        else:
            target_bands = [band for band in epoch.bands if band.casefold() in wanted]
            if not target_bands:
                continue
        epoch_products: list[tuple[TimelineEpoch, TimelineProduct]] = []
        for band in target_bands:
            candidates = [
                product
                for product in epoch.products
                if product.band == band
                and product.observation_id is not None
                and product.processing_status == "ready"
                and product.access_url_trusted
            ]
            if not candidates:
                continue
            candidates.sort(key=_product_priority)
            epoch_products.append((epoch, candidates[0]))
        if epoch_products:
            selected.extend(epoch_products)
            selected_epochs += 1
    return selected


def _product_priority(product: TimelineProduct) -> tuple[int, int, str, str]:
    return (
        0 if product.dataproduct_type == "image" else 1,
        0 if (product.dataproduct_subtype or "").casefold() == "science" else 1,
        -(product.calib_level or 0),
        product.obs_publisher_did or "",
    )


def _cache_key(
    request: TargetLightCurveRequest, cutout_size: float, photometry: PhotometryConfig
) -> str:
    payload: dict[str, Any] = {
        "request": request.model_dump(exclude={"photometry"}),
        "cutout_size": cutout_size,
        "photometry": photometry.model_dump(),
        "version": _LIGHTCURVE_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


__all__ = ["TargetLightCurveService", "build_light_curves"]
