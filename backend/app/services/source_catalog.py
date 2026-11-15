"""Persistent source catalogue and per-source light-curve/variability views.

Source ids are internal association ids. Persisting them lets a user move from a
target-level light-curve response to a stable source record without re-running
archive retrieval. Every record retains the query, target and processing
provenance that produced it.
"""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path

from app.core.cache import CacheStore
from app.core.config import Settings
from app.core.exceptions import SourceNotFoundError, SourceStorageError
from app.schemas.lightcurve import (
    AssociatedSource,
    AssociationConfig,
    BandVariability,
    LightCurveSeries,
    SourceDetailResponse,
    SourceLightCurve,
    SourceLightCurveResponse,
    SourceRecord,
    SourceVariability,
    SourceVariabilityResponse,
    TargetLightCurveResponse,
    TargetVariabilityResponse,
)

log = logging.getLogger(__name__)

_SOURCE_ID = re.compile(r"^[0-9a-f]{32}$")


class SourceCatalogService:
    def __init__(self, settings: Settings, cache: CacheStore) -> None:
        self.settings = settings
        self.cache = cache

    def store(
        self,
        response: TargetLightCurveResponse,
        associated: list[AssociatedSource],
        config: AssociationConfig,
    ) -> None:
        curves = {curve.source_id: curve for curve in response.sources}
        for source in associated:
            curve = curves.get(source.source_id)
            if curve is None:
                curve = _below_min_epochs_curve(source)
            record = SourceRecord(
                source=source,
                lightcurve=curve,
                target=response.target,
                query=response.query,
                provenance=response.provenance,
            )
            try:
                self._write(
                    self.settings.sources_dir / f"{source.source_id}.json", record
                )
                self._write(
                    self.settings.lightcurves_dir / f"{source.source_id}.json", curve
                )
            except OSError as exc:
                log.warning(
                    "source_record_write_failed",
                    extra={"source_id": source.source_id, "error": str(exc)},
                )
                continue
            self.cache.sources.set(source.source_id, record)
            self.cache.variability.set(
                source.source_id, source_variability(source, curve)
            )

    def source(self, source_id: str) -> SourceDetailResponse:
        record = self._record(source_id)
        return SourceDetailResponse(
            source=record.source,
            target=record.target,
            query=record.query,
            provenance=record.provenance,
        )

    def lightcurve(self, source_id: str) -> SourceLightCurveResponse:
        record = self._record(source_id)
        return SourceLightCurveResponse(
            source=record.lightcurve,
            target=record.target,
            query=record.query,
            provenance=record.provenance,
        )

    def variability(self, source_id: str) -> SourceVariabilityResponse:
        record = self._record(source_id)
        cached = self.cache.variability.get(source_id)
        if cached is None:
            cached = source_variability(record.source, record.lightcurve)
            self.cache.variability.set(source_id, cached)
        return SourceVariabilityResponse(
            source=cached,
            target=record.target,
            query=record.query,
            provenance=record.provenance,
        )

    def target_variability(
        self, response: TargetLightCurveResponse
    ) -> TargetVariabilityResponse:
        return TargetVariabilityResponse(
            target=response.target,
            query=response.query,
            epochs_considered=response.epochs_considered,
            epochs_processed=response.epochs_processed,
            products_processed=response.products_processed,
            sources=[source_variability_from_curve(curve) for curve in response.sources],
            failures=response.failures,
            association=response.association,
            warnings=response.warnings,
            provenance=response.provenance,
        )

    def _record(self, source_id: str) -> SourceRecord:
        if not _SOURCE_ID.fullmatch(source_id):
            raise SourceNotFoundError(
                f"Source {source_id!r} is not a valid internal source id."
            )
        cached = self.cache.sources.get(source_id)
        if isinstance(cached, SourceRecord):
            return cached
        path = self.settings.sources_dir / f"{source_id}.json"
        if not path.is_file():
            raise SourceNotFoundError(f"Source {source_id!r} was not found.")
        try:
            record = SourceRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SourceStorageError(
                f"Stored source record {source_id!r} could not be read.",
                details={"source_id": source_id},
            ) from exc
        if record.source.source_id != source_id:
            raise SourceStorageError(
                f"Stored source record {source_id!r} has a mismatched id.",
                details={"source_id": source_id},
            )
        self.cache.sources.set(source_id, record)
        return record

    def _write(self, path: Path, record) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(record.model_dump_json(), encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def source_variability(
    source: AssociatedSource, curve: SourceLightCurve
) -> SourceVariability:
    return SourceVariability(
        source_id=source.source_id,
        reference_ra_deg=source.reference_ra_deg,
        reference_dec_deg=source.reference_dec_deg,
        epoch_count=source.epoch_count,
        bands=source.bands,
        position_scatter_arcsec=source.position_scatter_arcsec,
        band_metrics=[_band_metrics(series) for series in curve.series],
        flags=curve.flags,
    )


def source_variability_from_curve(curve: SourceLightCurve) -> SourceVariability:
    return SourceVariability(
        source_id=curve.source_id,
        reference_ra_deg=curve.reference_ra_deg,
        reference_dec_deg=curve.reference_dec_deg,
        epoch_count=curve.epoch_count,
        bands=curve.bands,
        position_scatter_arcsec=curve.position_scatter_arcsec,
        band_metrics=[_band_metrics(series) for series in curve.series],
        flags=curve.flags,
    )


def _band_metrics(series: LightCurveSeries) -> BandVariability:
    return BandVariability(
        band=series.band,
        expected_epochs=series.expected_epochs,
        non_detections=series.non_detections,
        metrics=series.metrics,
    )


def _below_min_epochs_curve(source: AssociatedSource) -> SourceLightCurve:
    return SourceLightCurve(
        source_id=source.source_id,
        reference_ra_deg=source.reference_ra_deg,
        reference_dec_deg=source.reference_dec_deg,
        epoch_count=source.epoch_count,
        bands=source.bands,
        position_scatter_arcsec=source.position_scatter_arcsec,
        series=[],
        flags=sorted(set(source.flags) | {"below_min_epochs", "no_light_curve"}),
    )


__all__ = [
    "SourceCatalogService",
    "source_variability",
    "source_variability_from_curve",
]
