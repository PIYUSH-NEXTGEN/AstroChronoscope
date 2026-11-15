"""Angular cross-epoch association with an explicit ambiguity audit trail.

The matcher is a transparent nearest-neighbour association on the sphere. It
never silently merges two detections that are both plausible counterparts: the
ambiguity is recorded and the source is flagged for review. A duplicate detection
inside an epoch that has already been matched is excluded with a reason instead of
forcing a second light-curve point at the same time.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

from astropy.coordinates import SkyCoord
from astropy import units as u

from app.schemas.lightcurve import (
    ASSOCIATION_ALGORITHM_VERSION,
    AssociatedSource,
    AssociationCandidate,
    AssociationConfig,
    AssociationSummary,
    EpochMeasurements,
    ExcludedDetection,
    MatchedDetection,
)
from app.schemas.photometry import FluxMeasurement


@dataclass
class _State:
    source_id: str
    x: float
    y: float
    z: float
    count: int = 0
    epochs: set[str] = field(default_factory=set)
    detections: list[MatchedDetection] = field(default_factory=list)
    bands: set[str] = field(default_factory=set)
    flags: set[str] = field(default_factory=set)
    ambiguity_count: int = 0
    position_sigma_arcsec: float | None = None

    @property
    def ra(self) -> float:
        return math.degrees(math.atan2(self.y, self.x)) % 360.0

    @property
    def dec(self) -> float:
        return math.degrees(math.atan2(self.z, math.hypot(self.x, self.y)))

    def add(
        self,
        measurement: FluxMeasurement,
        epoch_key: str,
        separation: float | None,
        reference: float | None,
        candidate_states: list[tuple[float, _State]] | None = None,
        position_sigma: float | None = None,
        effective_match_radius: float | None = None,
    ) -> None:
        unit = _unit_vector(float(measurement.ra_deg), float(measurement.dec_deg))
        self.x += unit[0]
        self.y += unit[1]
        self.z += unit[2]
        self.count += 1
        self.epochs.add(epoch_key)
        if measurement.band:
            self.bands.add(measurement.band)
        if position_sigma is not None:
            self.position_sigma_arcsec = max(self.position_sigma_arcsec or 0.0, position_sigma)
        self.detections.append(
            MatchedDetection(
                measurement=measurement,
                epoch_key=epoch_key,
                separation_arcsec=_round(separation),
                reference_separation_arcsec=_round(reference),
                position_sigma_arcsec=_round(position_sigma),
                effective_match_radius_arcsec=_round(effective_match_radius),
                candidates=[
                    AssociationCandidate(
                        source_id=state.source_id,
                        separation_arcsec=_round(distance) or 0.0,
                        selected=state.source_id == self.source_id,
                    )
                    for distance, state in candidate_states or []
                ],
                flags=[],
            )
        )


def associate_epochs(
    epochs: list[EpochMeasurements],
    config: AssociationConfig,
) -> tuple[list[AssociatedSource], AssociationSummary]:
    ordered = sorted(
        epochs,
        key=lambda item: (item.time_mjd if item.time_mjd is not None else math.inf, item.epoch_key),
    )
    states: list[_State] = []
    excluded: list[ExcludedDetection] = []
    total_detections = 0
    ambiguous = 0
    duplicates = 0
    missing = 0

    for epoch in ordered:
        used_sources: set[str] = set()
        measurements = sorted(
            epoch.measurements,
            key=lambda item: -(item.snr if item.snr is not None else -math.inf),
        )
        for measurement in measurements:
            total_detections += 1
            if measurement.ra_deg is None or measurement.dec_deg is None:
                missing += 1
                excluded.append(
                    ExcludedDetection(
                        detection_id=measurement.detection_id,
                        observation_id=measurement.observation_id,
                        epoch_key=epoch.epoch_key,
                        reason="missing_position",
                    )
                )
                continue

            candidates = _candidates(measurement, states, config)
            audit_radius = (
                config.match_radius_arcsec
                + config.ambiguity_margin_arcsec
                + config.max_uncertainty_boost_arcsec
            )
            if not candidates:
                state = _new_state(measurement, epoch.epoch_key)
                states.append(state)
                used_sources.add(state.source_id)
                continue

            available = [item for item in candidates if item[1].source_id not in used_sources]
            if not available:
                _distance, state = candidates[0]
                if _already_seen(measurement.detection_id, state, epoch.epoch_key):
                    duplicates += 1
                    state.flags.add("duplicate_detection_in_epoch")
                    excluded.append(
                        ExcludedDetection(
                            detection_id=measurement.detection_id,
                            observation_id=measurement.observation_id,
                            epoch_key=epoch.epoch_key,
                            reason="duplicate_same_epoch",
                        )
                    )
                    continue
                # A distinct detection close to an already-matched source in the
                # same epoch is kept as its own source, with the crowding flagged.
                audit = _candidates(
                    measurement,
                    states,
                    config,
                    audit_radius,
                )
                new_state = _new_state(measurement, epoch.epoch_key, audit)
                new_state.flags.add("same_epoch_close_pair")
                states.append(new_state)
                used_sources.add(new_state.source_id)
                ambiguous += 1
                continue

            distance, state = available[0]
            reference = _separation(measurement.ra_deg, measurement.dec_deg, state.ra, state.dec)
            if _has_ambiguity(measurement, states, config, state.source_id):
                ambiguous += 1
                state.ambiguity_count += 1
                state.flags.add("ambiguous_match")
            audit = _candidates(
                measurement,
                states,
                config,
                audit_radius,
            )
            sigma = _combined_sigma(measurement, state)
            used_sources.add(state.source_id)
            state.add(
                measurement,
                epoch.epoch_key,
                distance,
                reference,
                audit,
                sigma,
                _match_limit(measurement, state, config),
            )

    sources = [_finalize(state, config) for state in states]
    sources.sort(
        key=lambda item: (-item.epoch_count, -item.detection_count, item.source_id)
    )
    sources_truncated = len(sources) > config.max_sources
    sources = sources[: config.max_sources]
    below_min_epoch_source_ids = [
        source.source_id for source in sources if source.epoch_count < config.min_epochs
    ]
    summary = AssociationSummary(
        match_radius_arcsec=config.match_radius_arcsec,
        input_epochs=len(epochs),
        input_detections=total_detections,
        source_count=len(sources),
        multi_epoch_sources=len(sources) - len(below_min_epoch_source_ids),
        below_min_epoch_sources=len(below_min_epoch_source_ids),
        below_min_epoch_source_ids=below_min_epoch_source_ids,
        sources_truncated=sources_truncated,
        ambiguous_matches=ambiguous,
        duplicates_dropped=duplicates,
        missing_position_dropped=missing,
        excluded=excluded,
    )
    return sources, summary


def _candidates(
    measurement: FluxMeasurement,
    states: list[_State],
    config: AssociationConfig,
    radius: float | None = None,
) -> list[tuple[float, _State]]:
    found: list[tuple[float, _State]] = []
    for state in states:
        limit = radius if radius is not None else _match_limit(measurement, state, config)
        separation = _separation(float(measurement.ra_deg), float(measurement.dec_deg), state.ra, state.dec)
        if separation <= limit:
            found.append((separation, state))
    found.sort(key=lambda item: item[0])
    return found


def _has_ambiguity(
    measurement: FluxMeasurement,
    states: list[_State],
    config: AssociationConfig,
    chosen_id: str,
) -> bool:
    count = 0
    for state in states:
        if state.source_id == chosen_id:
            continue
        limit = _match_limit(measurement, state, config) + config.ambiguity_margin_arcsec
        separation = _separation(float(measurement.ra_deg), float(measurement.dec_deg), state.ra, state.dec)
        if separation <= limit:
            count += 1
    return count > 0


def _measurement_sigma(measurement: FluxMeasurement) -> float | None:
    value = measurement.centroid_error_arcsec
    return value if value is not None and math.isfinite(value) and value > 0 else None


def _combined_sigma(measurement: FluxMeasurement, state: _State) -> float | None:
    values = [
        value
        for value in (_measurement_sigma(measurement), state.position_sigma_arcsec)
        if value is not None
    ]
    if not values:
        return None
    return math.sqrt(sum(value * value for value in values))


def _match_limit(
    measurement: FluxMeasurement, state: _State, config: AssociationConfig
) -> float:
    sigma = _combined_sigma(measurement, state)
    if sigma is None:
        return config.match_radius_arcsec
    boost = min(
        config.position_uncertainty_sigma * sigma,
        config.max_uncertainty_boost_arcsec,
    )
    return config.match_radius_arcsec + boost


def _already_seen(detection_id: str, state: _State, epoch_key: str) -> bool:
    return any(
        item.measurement.detection_id == detection_id and item.epoch_key == epoch_key
        for item in state.detections
    )


def _new_state(
    measurement: FluxMeasurement,
    epoch_key: str,
    candidates: list[tuple[float, _State]] | None = None,
) -> _State:
    source_id = hashlib.sha256(
        f"{ASSOCIATION_ALGORITHM_VERSION}|{epoch_key}|{measurement.detection_id}".encode("utf-8")
    ).hexdigest()[:32]
    state = _State(source_id=source_id, x=0.0, y=0.0, z=0.0)
    state.add(
        measurement,
        epoch_key,
        0.0,
        0.0,
        candidates,
        _measurement_sigma(measurement),
        None,
    )
    return state


def _finalize(state: _State, config: AssociationConfig) -> AssociatedSource:
    if state.count < 2:
        state.flags.add("single_epoch")
    if state.ambiguity_count:
        state.flags.add("ambiguous_match")
    scatters = [
        _separation(
            float(item.measurement.ra_deg), float(item.measurement.dec_deg), state.ra, state.dec
        )
        for item in state.detections
    ]
    scatter = (
        math.sqrt(sum(value ** 2 for value in scatters) / len(scatters)) if scatters else None
    )
    if scatter is not None and scatter > config.match_radius_arcsec:
        state.flags.add("large_position_scatter")
    if any(item.measurement.band is None for item in state.detections):
        state.flags.add("missing_band")
    if any(_measurement_sigma(item.measurement) is None for item in state.detections):
        state.flags.add("position_uncertainty_incomplete")
    return AssociatedSource(
        source_id=state.source_id,
        reference_ra_deg=_round(state.ra, 8),
        reference_dec_deg=_round(state.dec, 8),
        detection_count=state.count,
        epoch_count=len(state.epochs),
        bands=sorted(state.bands),
        detections=sorted(
            state.detections,
            key=lambda item: (item.measurement.time_mjd if item.measurement.time_mjd is not None else math.inf, item.measurement.detection_id),
        ),
        position_scatter_arcsec=_round(scatter),
        ambiguity_count=state.ambiguity_count,
        flags=sorted(state.flags),
    )


def _unit_vector(ra: float, dec: float) -> tuple[float, float, float]:
    ra_r, dec_r = math.radians(ra), math.radians(dec)
    return (
        math.cos(dec_r) * math.cos(ra_r),
        math.cos(dec_r) * math.sin(ra_r),
        math.sin(dec_r),
    )


def _separation(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    first = SkyCoord(ra1 * u.deg, dec1 * u.deg, frame="icrs")
    second = SkyCoord(ra2 * u.deg, dec2 * u.deg, frame="icrs")
    return float(first.separation(second).to_value(u.arcsec))


def _round(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None and math.isfinite(value) else None


__all__ = ["associate_epochs"]
