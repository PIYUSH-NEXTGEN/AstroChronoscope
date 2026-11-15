"""Phase 3 cross-epoch association, light-curve and variability contracts.

Scientific boundaries
---------------------
* A source id is an *internal* association id, not an official SPHEREx or
  external-catalogue identity.
* Light curves are split by band; incompatible bands are never concatenated.
* Metrics describe variability evidence. They do not classify an object as a
  variable, transient or candidate.
* Non-detections are counted, never converted to zero flux or fake upper limits.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from app.schemas.common import StrictModel, Target
from app.schemas.photometry import FluxMeasurement, PhotometryConfig
from app.schemas.sources import DetectionConfig, SourceQuality

ASSOCIATION_ALGORITHM_VERSION = "phase3-angular-association-v1"
VARIABILITY_ALGORITHM_VERSION = "phase3-variability-metrics-v1"


class AssociationConfig(StrictModel):
    match_radius_arcsec: float = Field(6.0, gt=0.0, le=120.0)
    ambiguity_margin_arcsec: float = Field(1.0, ge=0.0, le=60.0)
    position_uncertainty_sigma: float = Field(
        3.0,
        ge=0.0,
        le=10.0,
        description="Multiplier on the combined centroid error when expanding the match radius",
    )
    max_uncertainty_boost_arcsec: float = Field(
        3.0, ge=0.0, le=30.0, description="Upper bound on the uncertainty-driven match-radius expansion"
    )
    min_epochs: int = Field(2, ge=1, le=100)
    max_sources: int = Field(100, ge=1, le=5000)


class EpochMeasurements(StrictModel):
    epoch_key: str
    observation_id: str
    obs_publisher_did: str | None = None
    band: str | None = None
    time_utc: str | None = None
    time_mjd: float | None = None
    measurements: list[FluxMeasurement] = Field(default_factory=list)
    quality: SourceQuality = "PASS"
    warnings: list[str] = Field(default_factory=list)


class AssociationCandidate(StrictModel):
    source_id: str
    separation_arcsec: float
    selected: bool = False


class MatchedDetection(StrictModel):
    measurement: FluxMeasurement
    epoch_key: str
    separation_arcsec: float | None = None
    reference_separation_arcsec: float | None = None
    position_sigma_arcsec: float | None = None
    effective_match_radius_arcsec: float | None = None
    candidates: list[AssociationCandidate] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class ExcludedDetection(StrictModel):
    detection_id: str
    observation_id: str
    epoch_key: str
    reason: Literal["duplicate_same_epoch", "missing_position"]


class AssociatedSource(StrictModel):
    source_id: str
    reference_ra_deg: float
    reference_dec_deg: float
    detection_count: int
    epoch_count: int
    bands: list[str] = Field(default_factory=list)
    detections: list[MatchedDetection] = Field(default_factory=list)
    position_scatter_arcsec: float | None = None
    ambiguity_count: int = 0
    flags: list[str] = Field(default_factory=list)
    association_method: str = "angular_nearest_neighbour"
    association_version: str = ASSOCIATION_ALGORITHM_VERSION


class AssociationSummary(StrictModel):
    method: str = "angular_nearest_neighbour"
    version: str = ASSOCIATION_ALGORITHM_VERSION
    match_radius_arcsec: float
    input_epochs: int
    input_detections: int
    source_count: int
    multi_epoch_sources: int = 0
    below_min_epoch_sources: int = 0
    below_min_epoch_source_ids: list[str] = Field(default_factory=list)
    sources_truncated: bool = False
    ambiguous_matches: int
    duplicates_dropped: int
    missing_position_dropped: int
    excluded: list[ExcludedDetection] = Field(default_factory=list)


class VariabilityMetrics(StrictModel):
    algorithm_version: str = VARIABILITY_ALGORITHM_VERSION
    status: Literal["computed", "insufficient_data", "undefined_uncertainties"]
    unit: Literal["Jy"] = "Jy"
    n_points: int
    mean_flux_jy: float | None = None
    median_flux_jy: float | None = None
    weighted_mean_flux_jy: float | None = None
    sample_std_jy: float | None = None
    mad_jy: float | None = None
    robust_sigma_jy: float | None = None
    fractional_variability: float | None = Field(
        None, description="sqrt(sample_variance - mean_squared_error) / mean_flux; null when undefined"
    )
    chi2: float | None = None
    dof: int | None = None
    reduced_chi2: float | None = None
    chi2_pvalue: float | None = None
    peak_to_peak_jy: float | None = None
    robust_range_jy: float | None = None
    min_flux_jy: float | None = None
    max_flux_jy: float | None = None
    temporal_baseline_days: float | None = None
    max_gap_days: float | None = None
    flags: list[str] = Field(default_factory=list)


class LightCurvePoint(StrictModel):
    time_utc: str | None = None
    time_mjd: float | None = None
    flux_jy: float
    flux_jy_error: float | None = None
    unit: Literal["Jy"] = "Jy"
    band: str | None = None
    observation_id: str
    obs_publisher_did: str | None = None
    detection_id: str
    snr: float | None = None
    quality: SourceQuality
    flags: list[str] = Field(default_factory=list)


class LightCurveSeries(StrictModel):
    band: str | None = None
    points: list[LightCurvePoint] = Field(default_factory=list)
    expected_epochs: int = 0
    non_detections: int = 0
    metrics: VariabilityMetrics


class SourceLightCurve(StrictModel):
    source_id: str
    reference_ra_deg: float
    reference_dec_deg: float
    epoch_count: int
    bands: list[str] = Field(default_factory=list)
    position_scatter_arcsec: float | None = None
    series: list[LightCurveSeries] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class TargetLightCurveRequest(StrictModel):
    ra: float = Field(..., ge=0.0, lt=360.0, allow_inf_nan=False)
    dec: float = Field(..., ge=-90.0, le=90.0, allow_inf_nan=False)
    radius_deg: float = Field(0.05, gt=0.0, allow_inf_nan=False, description="Archive search radius")
    cutout_size_deg: float = Field(0.1, gt=0.0, allow_inf_nan=False, description="Cutout size around the target")
    collection: str | None = None
    bands: list[str] | None = None
    start_date: str | None = None
    end_date: str | None = None
    max_observations: int = Field(20, ge=1, le=200)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    photometry: PhotometryConfig | None = None
    association: AssociationConfig = Field(default_factory=AssociationConfig)


class EpochFailure(StrictModel):
    epoch_key: str | None = None
    observation_id: str | None = None
    band: str | None = None
    code: str
    message: str


class LightCurveQuery(StrictModel):
    radius_deg: float
    cutout_size_deg: float
    collection: str | None = None
    bands: list[str] | None = None
    start_date: str | None = None
    end_date: str | None = None


class LightCurveProvenance(StrictModel):
    created_at: str
    software: dict[str, str]
    association: str = ASSOCIATION_ALGORITHM_VERSION
    variability: str = VARIABILITY_ALGORITHM_VERSION


class TargetLightCurveResponse(StrictModel):
    success: Literal[True] = True
    target: Target
    query: LightCurveQuery
    epochs_considered: int
    epochs_processed: int
    products_processed: int
    sources: list[SourceLightCurve] = Field(default_factory=list)
    failures: list[EpochFailure] = Field(default_factory=list)
    association: AssociationSummary
    warnings: list[str] = Field(default_factory=list)
    provenance: LightCurveProvenance


class SourceRecord(StrictModel):
    source: AssociatedSource
    lightcurve: SourceLightCurve
    target: Target
    query: LightCurveQuery
    provenance: LightCurveProvenance


class SourceDetailResponse(StrictModel):
    success: Literal[True] = True
    source: AssociatedSource
    target: Target
    query: LightCurveQuery
    provenance: LightCurveProvenance


class SourceLightCurveResponse(StrictModel):
    success: Literal[True] = True
    source: SourceLightCurve
    target: Target
    query: LightCurveQuery
    provenance: LightCurveProvenance


class BandVariability(StrictModel):
    band: str | None = None
    expected_epochs: int
    non_detections: int
    metrics: VariabilityMetrics


class SourceVariability(StrictModel):
    source_id: str
    reference_ra_deg: float
    reference_dec_deg: float
    epoch_count: int
    bands: list[str] = Field(default_factory=list)
    position_scatter_arcsec: float | None = None
    band_metrics: list[BandVariability] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class SourceVariabilityResponse(StrictModel):
    success: Literal[True] = True
    source: SourceVariability
    target: Target
    query: LightCurveQuery
    provenance: LightCurveProvenance


class TargetVariabilityResponse(StrictModel):
    success: Literal[True] = True
    target: Target
    query: LightCurveQuery
    epochs_considered: int
    epochs_processed: int
    products_processed: int
    sources: list[SourceVariability] = Field(default_factory=list)
    failures: list[EpochFailure] = Field(default_factory=list)
    association: AssociationSummary
    warnings: list[str] = Field(default_factory=list)
    provenance: LightCurveProvenance


__all__ = [
    "ASSOCIATION_ALGORITHM_VERSION",
    "AssociationCandidate",
    "AssociationConfig",
    "AssociationSummary",
    "AssociatedSource",
    "BandVariability",
    "EpochFailure",
    "EpochMeasurements",
    "ExcludedDetection",
    "LightCurvePoint",
    "LightCurveProvenance",
    "LightCurveQuery",
    "LightCurveSeries",
    "MatchedDetection",
    "SourceDetailResponse",
    "SourceLightCurve",
    "SourceLightCurveResponse",
    "SourceRecord",
    "SourceVariability",
    "SourceVariabilityResponse",
    "TargetLightCurveRequest",
    "TargetLightCurveResponse",
    "TargetVariabilityResponse",
    "VARIABILITY_ALGORITHM_VERSION",
    "VariabilityMetrics",
]
