"""Phase 3 source-detection request/response contracts.

Vocabulary
----------
``PASS``
    The product was prepared and every required input was available.
``WARNING``
    The result is usable but carries a documented caveat (edge source, blended
    source, missing variance, no sources found, ...).
``FAIL``
    The image could not be used (for example no valid pixels).
``NOT_VERIFIED``
    A required quantity (background or noise) could not be estimated, so no
    detection was asserted. This is never treated as ``PASS``.

No field ever fabricates a value: missing measurements are ``None`` and a
non-detection is not zero flux.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from app.schemas.common import StrictModel, Target

SourceQuality = Literal["PASS", "WARNING", "FAIL", "NOT_VERIFIED"]

#: Version of the detection algorithm. Bumped whenever numerics change so that
#: cached catalogues from a different algorithm are not reused.
DETECTION_ALGORITHM_VERSION = "phase3-source-detection-v1"


class DetectionConfig(StrictModel):
    """Tunable detection parameters.

    Defaults are a starting point, not a universal truth: crowded fields,
    extended sources and structured backgrounds require re-tuning and
    documented tests.
    """

    threshold_sigma: float = Field(
        5.0, ge=1.0, le=20.0, description="Detection threshold in units of the smoothed background noise"
    )
    min_area_px: int = Field(5, ge=1, le=10000, description="Minimum connected area in pixels")
    max_sources: int = Field(500, ge=1, le=10000, description="Maximum detections returned (brightest first)")
    filter_fwhm_px: float = Field(
        1.5, ge=0.5, le=20.0, description="FWHM of the Gaussian matched filter in pixels"
    )
    edge_margin_px: int = Field(2, ge=0, le=100, description="Centres closer than this to the border are edge sources")
    background_mesh: int = Field(4, ge=1, le=16, description="Mesh cells per axis for the background-variation check")
    deblend: bool = Field(True, description="Split a connected component that contains several local maxima")


class DetectionRequest(StrictModel):
    observation_id: str = Field(..., min_length=1, max_length=1024)
    ra: float | None = Field(None, ge=0.0, lt=360.0, allow_inf_nan=False)
    dec: float | None = Field(None, ge=-90.0, le=90.0, allow_inf_nan=False)
    size: float | None = Field(None, gt=0.0, allow_inf_nan=False, description="Cutout size in degrees")
    config: DetectionConfig = Field(default_factory=DetectionConfig)


class BackgroundEstimate(StrictModel):
    method: Literal["sigma_clipped_median", "none"] = "sigma_clipped_median"
    level: float | None = None
    sigma: float | None = None
    mesh_variation: float | None = None
    reliable: bool = False
    valid_pixels: int = 0
    note: str | None = None


class ImagePreparationSummary(StrictModel):
    shape: list[int]
    science_extname: str
    bunit: str | None = None
    valid_pixels: int
    invalid_pixels: int
    masked_pixels: int
    nan_pixels: int
    inf_pixels: int
    valid_fraction: float
    variance_available: bool
    variance_valid_fraction: float | None = None
    pixel_scale_arcsec: float | None = None
    pixel_solid_angle_sr: float | None = None
    pixel_solid_angle_model: Literal["projection_plane_constant", "unavailable"] = "unavailable"
    psf_fwhm_arcsec: float | None = None
    mask_bits: dict[str, int] = Field(default_factory=dict)


class DetectedSource(StrictModel):
    detection_id: str
    source_id: str | None = Field(
        None, description="Internal source id assigned by cross-epoch association; null for a single-image detection"
    )
    x: float = Field(..., description="Flux-weighted centroid, 0-based pixel x")
    y: float = Field(..., description="Flux-weighted centroid, 0-based pixel y")
    ra_deg: float | None = None
    dec_deg: float | None = None
    peak_value: float | None = Field(None, description="Peak native pixel value, not a flux")
    source_area: int = Field(..., description="Number of pixels in the segmented detection")
    centroid_x: float
    centroid_y: float
    detection_snr: float | None = Field(None, description="Background-subtracted pixel sum over noise*sqrt(area)")
    centroid_error_px: float | None = Field(
        None, description="Approximate 1-sigma centroid error, FWHM/(2.355*SNR); not an astrometric calibration error"
    )
    centroid_error_arcsec: float | None = None
    fwhm_px: float | None = None
    fwhm_arcsec: float | None = None
    ellipticity: float | None = None
    orientation_deg: float | None = None
    background_level: float | None = Field(
        None, description="Global background level used for every detection in this image"
    )
    noise_sigma: float | None = Field(
        None, description="Global background sigma used for this detection; a per-source local noise is not yet computed"
    )
    blended: bool = False
    flags: list[str] = Field(default_factory=list)
    quality: SourceQuality


class DetectionQuality(StrictModel):
    overall_status: SourceQuality
    warnings: list[str] = Field(default_factory=list)


class DetectionProvenance(StrictModel):
    created_at: str
    software: dict[str, str]
    algorithm: str
    algorithm_version: str = DETECTION_ALGORITHM_VERSION
    observation_id: str
    obs_publisher_did: str | None = None
    band: str | None = None
    observation_time: str | None = None
    fits_sha256: str
    cache_key: str
    config: dict[str, Any]


class DetectionResponse(StrictModel):
    success: Literal[True] = True
    observation_id: str
    source: Literal["cutout", "product"]
    source_url: str | None = None
    target: Target | None = None
    size_deg: float | None = None
    config: DetectionConfig
    image: ImagePreparationSummary
    background: BackgroundEstimate
    detections: list[DetectedSource]
    total_detections: int
    truncated: bool = False
    quality: DetectionQuality
    provenance: DetectionProvenance


__all__ = [
    "BackgroundEstimate",
    "DETECTION_ALGORITHM_VERSION",
    "DetectedSource",
    "DetectionConfig",
    "DetectionProvenance",
    "DetectionQuality",
    "DetectionRequest",
    "DetectionResponse",
    "ImagePreparationSummary",
    "SourceQuality",
]
