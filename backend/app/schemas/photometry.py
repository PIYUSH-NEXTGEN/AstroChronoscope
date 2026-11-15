"""Phase 3 aperture-photometry contracts.

Measurement policy
------------------
SPHEREx level-2 ``IMAGE`` pixels store surface brightness (normally MJy/sr), not
integrated flux. An aperture sum in MJy/sr becomes a flux only after multiplying
by the solid angle each pixel subtends. The conversion is therefore explicit and
recorded (``solid_angle_sr``); no aperture correction or PSF correction is
applied in this baseline and the omission is recorded on every measurement.

Missing or invalid pixels reduce the aperture's valid coverage; they are never
treated as zero. A negative background-subtracted flux is kept, not clipped.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from app.schemas.common import StrictModel, Target
from app.schemas.sources import (
    BackgroundEstimate,
    DetectedSource,
    DetectionConfig,
    DetectionProvenance,
    ImagePreparationSummary,
    SourceQuality,
)

PHOTOMETRY_ALGORITHM_VERSION = "phase3-aperture-photometry-v1"


class PhotometryConfig(StrictModel):
    aperture_fwhm_factor: float = Field(3.0, ge=1.0, le=10.0)
    annulus_inner_factor: float = Field(1.3, ge=1.0, le=5.0)
    annulus_outer_factor: float = Field(2.0, ge=1.1, le=10.0)
    aperture_oversample: int = Field(8, ge=1, le=32)
    min_valid_fraction: float = Field(0.5, ge=0.0, le=1.0)
    psf_fwhm_fallback_px: float = Field(3.0, gt=0.0, le=50.0)
    min_annulus_pixels: int = Field(9, ge=1, le=10000)
    min_annulus_fraction: float = Field(0.25, ge=0.0, le=1.0)


class PhotometryRequest(StrictModel):
    observation_id: str = Field(..., min_length=1, max_length=1024)
    ra: float | None = Field(None, ge=0.0, lt=360.0, allow_inf_nan=False)
    dec: float | None = Field(None, ge=-90.0, le=90.0, allow_inf_nan=False)
    size: float | None = Field(None, gt=0.0, allow_inf_nan=False, description="Cutout size in degrees")
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    config: PhotometryConfig | None = None


class FluxMeasurement(StrictModel):
    detection_id: str
    observation_id: str
    obs_publisher_did: str | None = None
    band: str | None = None
    observation_time: str | None = None
    time_mjd: float | None = None

    x: float
    y: float
    ra_deg: float | None = None
    dec_deg: float | None = None
    centroid_error_px: float | None = Field(
        None, description="Approximate 1-sigma centroid error inherited from detection, FWHM/(2.355*SNR)"
    )
    centroid_error_arcsec: float | None = None

    aperture_radius_px: float
    aperture_radius_arcsec: float | None = None
    aperture_area_px: float
    annulus_area_px: float | None = None
    aperture_valid_fraction: float
    psf_fwhm_px: float
    fwhm_source: Literal["psf", "fallback"]

    aperture_sum: float | None = Field(None, description="Background-subtracted aperture sum in native units")
    aperture_sum_unit: str | None = Field(None, description="Native BUNIT of aperture_sum, if known")
    background_level: float | None = None
    background_sigma: float | None = None
    background_method: Literal["annulus_sigma_clipped", "global", "unavailable"] = "unavailable"

    solid_angle_sr: float | None = None
    flux_mjy: float | None = Field(None, description="Integrated flux density in MJy, when units permit")
    flux_jy: float | None = Field(None, description="Integrated flux density in Jy, when units permit")
    flux_jy_error: float | None = None
    snr: float | None = Field(None, description="flux_jy / flux_jy_error; not a variability significance")
    uncertainty_method: Literal["variance_array", "background_sigma", "mixed", "unavailable"] = "unavailable"

    aperture_correction_applied: Literal[False] = False
    measurement_method: str = "circular_aperture"
    flags: list[str] = Field(default_factory=list)
    quality: SourceQuality


class PhotometryProvenance(StrictModel):
    created_at: str
    software: dict[str, str]
    method: str
    algorithm_version: str = PHOTOMETRY_ALGORITHM_VERSION
    detection_algorithm_version: str
    observation_id: str
    obs_publisher_did: str | None = None
    band: str | None = None
    observation_time: str | None = None
    fits_sha256: str
    cache_key: str
    config: dict[str, Any]


class PhotometryResponse(StrictModel):
    success: Literal[True] = True
    observation_id: str
    source: Literal["cutout", "product"]
    source_url: str | None = None
    target: Target | None = None
    config: PhotometryConfig
    detection: DetectionProvenance
    image: ImagePreparationSummary
    background: BackgroundEstimate
    detections: list[DetectedSource]
    measurements: list[FluxMeasurement]
    total_measurements: int
    quality: SourceQuality
    warnings: list[str] = Field(default_factory=list)
    provenance: PhotometryProvenance


__all__ = [
    "FluxMeasurement",
    "PHOTOMETRY_ALGORITHM_VERSION",
    "PhotometryConfig",
    "PhotometryProvenance",
    "PhotometryRequest",
    "PhotometryResponse",
]
