"""Phase 3 aperture photometry on each product's native WCS grid.

The measurement is deliberately simple and auditable:

* circular aperture with fractional-pixel coverage from sub-pixel sampling;
* local background from a sigma-clipped annulus, falling back to the global
  background when the annulus is unusable;
* uncertainty from the VARIANCE extension when available, otherwise from the
  background sigma, plus the background-estimation term;
* surface brightness (MJy/sr) is multiplied by the pixel solid angle before it
  becomes an integrated flux. Unsupported units produce no Jy value instead of a
  guessed one.

No aperture correction, PSF correction, deblending correction or image
subtraction is applied. Those limitations are explicit in the output.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from astropy.stats import sigma_clipped_stats

from app.core.cache import CacheStore
from app.core.config import Settings
from app.core.versions import software_versions
from app.schemas.common import Target
from app.schemas.photometry import (
    PHOTOMETRY_ALGORITHM_VERSION,
    FluxMeasurement,
    PhotometryConfig,
    PhotometryProvenance,
    PhotometryRequest,
    PhotometryResponse,
)
from app.schemas.sources import (
    DETECTION_ALGORITHM_VERSION,
    DetectedSource,
    DetectionRequest,
    DetectionResponse,
    SourceQuality,
)
from app.services.artifact_resolver import resolve_observation_artifact
from app.services.image_preparation import PreparedImage, prepare_image
from app.services.observation_service import ObservationService
from app.services.source_detection import SourceDetectionService

log = logging.getLogger(__name__)

_SURFACE_BRIGHTNESS_UNITS = {"mjy/sr", "mjy*sr-1", "mjysr-1"}
_INTEGRATED_UNITS = {"mjy"}


class PhotometryService:
    def __init__(
        self,
        observations: ObservationService,
        detection: SourceDetectionService | None = None,
    ) -> None:
        self.observations = observations
        self.settings: Settings = observations.settings
        self.cache: CacheStore = observations.cache
        self.detection = detection or SourceDetectionService(observations)
        self.settings.ensure_dirs()

    # ------------------------------------------------------------------ public
    def measure(self, request: PhotometryRequest) -> PhotometryResponse:
        detection_request = DetectionRequest(
            observation_id=request.observation_id,
            ra=request.ra,
            dec=request.dec,
            size=request.size,
            config=request.detection,
        )
        detection_response = self.detection.detect(detection_request)
        return self.measure_detection(detection_response, request.config)

    def measure_detection(
        self,
        detection_response: DetectionResponse,
        config: PhotometryConfig | None = None,
    ) -> PhotometryResponse:
        config = config or self.config_from_settings()
        key = _cache_key(detection_response, config)
        cached = self.cache.photometry.get(key)
        if cached is not None:
            return cached

        resolved = resolve_observation_artifact(
            self.observations,
            detection_response.observation_id,
            detection_response.target.ra if detection_response.target else None,
            detection_response.target.dec if detection_response.target else None,
            detection_response.size_deg if detection_response.target else None,
            default_size=self.settings.comparison_size_deg,
        )
        prepared = prepare_image(resolved.path, mesh=detection_response.config.background_mesh)
        context = ObservationContext(
            observation_id=detection_response.observation_id,
            obs_publisher_did=resolved.detail.obs_publisher_did,
            band=resolved.detail.band,
            observation_time=resolved.detail.start_time,
            time_mjd=resolved.detail.t_min_mjd,
        )
        measurements = measure_prepared(prepared, detection_response.detections, config, context=context)
        warnings = sorted({flag for item in measurements for flag in item.flags})
        quality = _overall_quality(detection_response.detections, measurements)

        response = PhotometryResponse(
            observation_id=detection_response.observation_id,
            source=resolved.source,
            source_url=detection_response.source_url,
            target=Target(ra=detection_response.target.ra, dec=detection_response.target.dec)
            if detection_response.target
            else None,
            config=config,
            detection=detection_response.provenance,
            image=detection_response.image,
            background=detection_response.background,
            detections=detection_response.detections,
            measurements=measurements,
            total_measurements=len(measurements),
            quality=quality,
            warnings=warnings,
            provenance=PhotometryProvenance(
                created_at=datetime.now(timezone.utc).isoformat(),
                software=software_versions(),
                method=(
                    "fractional-pixel circular aperture on the native WCS grid; "
                    "sigma-clipped annulus background; propagated uncertainty; "
                    "surface brightness integrated with the pixel solid angle"
                ),
                detection_algorithm_version=DETECTION_ALGORITHM_VERSION,
                observation_id=detection_response.observation_id,
                obs_publisher_did=detection_response.provenance.obs_publisher_did,
                band=detection_response.provenance.band,
                observation_time=detection_response.provenance.observation_time,
                fits_sha256=detection_response.provenance.fits_sha256,
                cache_key=key,
                config=config.model_dump(),
            ),
        )
        self.cache.photometry.set(key, response)
        log.info(
            "photometry_completed",
            extra={
                "observation_id": detection_response.observation_id,
                "measurements": len(measurements),
                "quality": quality,
            },
        )
        return response

    def config_from_settings(self) -> PhotometryConfig:
        return PhotometryConfig(
            aperture_fwhm_factor=self.settings.photometry_aperture_fwhm_factor,
            annulus_inner_factor=self.settings.photometry_annulus_inner_factor,
            annulus_outer_factor=self.settings.photometry_annulus_outer_factor,
            aperture_oversample=self.settings.photometry_aperture_oversample,
            min_valid_fraction=self.settings.photometry_min_valid_fraction,
        )


# ------------------------------------------------------------------- algorithm
@dataclass(frozen=True)
class ObservationContext:
    observation_id: str
    obs_publisher_did: str | None = None
    band: str | None = None
    observation_time: str | None = None
    time_mjd: float | None = None


def measure_prepared(
    prepared: PreparedImage,
    detections: list[DetectedSource],
    config: PhotometryConfig,
    *,
    context: ObservationContext | None = None,
) -> list[FluxMeasurement]:
    context = context or ObservationContext(observation_id="unknown")
    fwhm_px, source = _psf_fwhm_px(prepared, config)
    radius = max(1.0, fwhm_px * config.aperture_fwhm_factor)
    geometry = _ApertureGeometry(prepared.shape, radius, config)
    return [
        _measure_one(prepared, detection, config, radius, fwhm_px, source, geometry, context)
        for detection in detections
    ]


class _ApertureGeometry:
    """Reusable full-frame aperture/annulus index helper for one radius."""

    def __init__(self, shape: tuple[int, int], radius: float, config: PhotometryConfig) -> None:
        self.shape = shape
        self.radius = radius
        inner = radius * config.annulus_inner_factor
        outer = radius * config.annulus_outer_factor
        self.inner = inner
        self.outer = outer
        self.oversample = config.aperture_oversample


def _measure_one(
    prepared: PreparedImage,
    detection: DetectedSource,
    config: PhotometryConfig,
    radius: float,
    fwhm_px: float,
    fwhm_source: str,
    geometry: _ApertureGeometry,
    context: ObservationContext,
) -> FluxMeasurement:
    cx = float(detection.centroid_x)
    cy = float(detection.centroid_y)
    ny, nx = prepared.shape
    oversample = config.aperture_oversample

    ap_weights, ap_slice = _coverage(cx, cy, 0.0, radius, (ny, nx), oversample)
    ann_weights, ann_slice = _coverage(
        cx, cy, geometry.inner, geometry.outer, (ny, nx), oversample
    )

    ap_area = float(ap_weights.sum())
    ann_area = float(ann_weights.sum())
    valid_local = prepared.valid[ap_slice]
    valid_area = float((ap_weights * valid_local).sum())
    valid_fraction = valid_area / ap_area if ap_area > 0 else 0.0

    flags: set[str] = set(detection.flags)
    if cx - radius < 0 or cy - radius < 0 or cx + radius > nx - 1 or cy + radius > ny - 1:
        flags.add("aperture_truncated")

    background, background_sigma, background_method, annulus_pixels = _background(
        prepared, ann_weights, ann_slice, config
    )
    if background_method == "unavailable":
        flags.add("background_unavailable")

    if valid_fraction < config.min_valid_fraction:
        flags.add("low_valid_coverage")

    bunit = prepared.bunit
    unit_mode = _unit_mode(bunit)
    if unit_mode is None:
        flags.add("unit_unsupported" if bunit else "missing_bunit")

    solid_angle = prepared.pixel_solid_angle_sr
    if unit_mode == "surface_brightness" and solid_angle is None:
        flags.add("missing_solid_angle")

    flux_native: float | None = None
    uncertainty_var: float | None = None
    uncertainty_method = "unavailable"

    if background is not None and valid_fraction >= config.min_valid_fraction and ap_area > 0:
        difference = prepared.data[ap_slice] - background
        weights_valid = np.where(valid_local, ap_weights, 0.0)
        flux_native = float(np.sum(weights_valid * difference))
        uncertainty_var, uncertainty_method = _variance(
            prepared,
            ap_slice,
            ap_weights,
            valid_local,
            background_sigma,
            annulus_pixels if background_method == "annulus_sigma_clipped" else None,
        )
    if uncertainty_var is None:
        flags.add("uncertainty_unavailable")

    flux_mjy, flux_jy, flux_jy_error, flux_flags = _convert_flux(
        flux_native, uncertainty_var, unit_mode, bunit, solid_angle
    )
    flags |= flux_flags
    if flux_jy is not None and flux_jy <= 0:
        flags.add("nonpositive_flux")
    if flux_mjy is not None and flux_jy_error is None:
        flags.add("uncertainty_unavailable")

    snr = (
        flux_jy / flux_jy_error
        if flux_jy is not None and flux_jy_error is not None and flux_jy_error > 0
        else None
    )

    if "background_unavailable" in flags:
        quality: SourceQuality = "NOT_VERIFIED"
    elif flux_native is None:
        quality = "FAIL"
    elif flux_jy is None or flags:
        quality = "WARNING"
    else:
        quality = "PASS"

    return FluxMeasurement(
        detection_id=detection.detection_id,
        observation_id=context.observation_id,
        obs_publisher_did=context.obs_publisher_did,
        band=context.band,
        observation_time=context.observation_time,
        time_mjd=context.time_mjd,
        x=round(cx, 4),
        y=round(cy, 4),
        ra_deg=detection.ra_deg,
        dec_deg=detection.dec_deg,
        centroid_error_px=detection.centroid_error_px,
        centroid_error_arcsec=detection.centroid_error_arcsec,
        aperture_radius_px=round(radius, 4),
        aperture_radius_arcsec=(
            radius * prepared.pixel_scale_arcsec if prepared.pixel_scale_arcsec is not None else None
        ),
        aperture_area_px=round(ap_area, 6),
        annulus_area_px=round(ann_area, 6),
        aperture_valid_fraction=round(valid_fraction, 6),
        psf_fwhm_px=round(fwhm_px, 4),
        fwhm_source=fwhm_source,
        aperture_sum=_round(flux_native),
        aperture_sum_unit=bunit,
        background_level=_round(background),
        background_sigma=_round(background_sigma),
        background_method=background_method,
        solid_angle_sr=_round(solid_angle),
        flux_mjy=_round(flux_mjy),
        flux_jy=_round(flux_jy),
        flux_jy_error=_round(flux_jy_error),
        snr=_round(snr),
        uncertainty_method=uncertainty_method,
        flags=sorted(flags),
        quality=quality,
    )


def _background(
    prepared: PreparedImage,
    annulus_weights: np.ndarray,
    annulus_slice: tuple[slice, slice],
    config: PhotometryConfig,
) -> tuple[float | None, float | None, str, int]:
    annulus_valid = (annulus_weights >= 0.5) & prepared.valid[annulus_slice]
    annulus_area = float(annulus_weights.sum())
    count = int(annulus_valid.sum())
    minimum = max(config.min_annulus_pixels, int(config.min_annulus_fraction * annulus_area))
    if count >= minimum:
        values = prepared.data[annulus_slice][annulus_valid]
        _, level, sigma = sigma_clipped_stats(values, sigma=3.0, maxiters=5)
        if math.isfinite(level) and math.isfinite(sigma) and sigma > 0:
            return float(level), float(sigma), "annulus_sigma_clipped", max(1, count)

    background = prepared.background
    if background.level is None or background.sigma is None or background.sigma <= 0:
        return None, None, "unavailable", 0
    return (
        float(background.level),
        float(background.sigma),
        "global",
        max(1, background.valid_pixels),
    )


def _variance(
    prepared: PreparedImage,
    aperture_slice: tuple[slice, slice],
    weights: np.ndarray,
    valid_local: np.ndarray,
    background_sigma: float | None,
    annulus_pixels: int | None,
) -> tuple[float | None, str]:
    sigma = background_sigma if background_sigma is not None and background_sigma > 0 else None
    if prepared.variance is None:
        if sigma is None:
            return None, "unavailable"
        var = float(np.sum((weights[valid_local] ** 2))) * sigma ** 2
        var += _background_term(weights, valid_local, sigma, annulus_pixels or prepared.background.valid_pixels)
        return var, "background_sigma"

    variance_local = prepared.variance[aperture_slice]
    finite = np.isfinite(variance_local) & valid_local
    missing = valid_local & ~finite
    var = float(np.sum((weights[finite] ** 2) * variance_local[finite]))
    method = "variance_array"
    if missing.any():
        if sigma is None:
            return None, "unavailable"
        var += float(np.sum((weights[missing] ** 2))) * sigma ** 2
        method = "mixed"
    if sigma is not None:
        var += _background_term(
            weights, valid_local, sigma, annulus_pixels or prepared.background.valid_pixels
        )
    return var, method


def _background_term(
    weights: np.ndarray, valid_local: np.ndarray, sigma: float, samples: int
) -> float:
    effective = max(1, int(samples))
    return float((weights[valid_local].sum() ** 2)) * sigma ** 2 / effective


def _convert_flux(
    flux_native: float | None,
    variance: float | None,
    unit_mode: str | None,
    bunit: str | None,
    solid_angle: float | None,
) -> tuple[float | None, float | None, float | None, set[str]]:
    flags: set[str] = set()
    if flux_native is None or unit_mode is None:
        return None, None, None, flags
    if unit_mode == "surface_brightness":
        if solid_angle is None:
            return None, None, None, flags
        flux_mjy = flux_native * solid_angle
        jansky = flux_mjy * 1e6
        error = math.sqrt(variance) * solid_angle * 1e6 if variance is not None else None
        return flux_mjy, jansky, error, flags
    if unit_mode == "integrated_mjy":
        flux_mjy = flux_native
        jansky = flux_mjy * 1e6
        error = math.sqrt(variance) * 1e6 if variance is not None else None
        return flux_mjy, jansky, error, flags
    flags.add("unit_unsupported")
    return None, None, None, flags


def _unit_mode(bunit: str | None) -> str | None:
    if not bunit:
        return None
    token = bunit.strip().casefold().replace(" ", "").replace("**", "").replace("^", "")
    if token in _SURFACE_BRIGHTNESS_UNITS:
        return "surface_brightness"
    if token in _INTEGRATED_UNITS:
        return "integrated_mjy"
    return None


def _psf_fwhm_px(prepared: PreparedImage, config: PhotometryConfig) -> tuple[float, str]:
    if prepared.psf_fwhm_arcsec is not None and prepared.pixel_scale_arcsec:
        return prepared.psf_fwhm_arcsec / prepared.pixel_scale_arcsec, "psf"
    return config.psf_fwhm_fallback_px, "fallback"


def _coverage(
    cx: float,
    cy: float,
    radius_inner: float,
    radius_outer: float,
    shape: tuple[int, int],
    oversample: int,
) -> tuple[np.ndarray, tuple[slice, slice]]:
    """Fractional pixel coverage of a disk/annulus, computed by sub-sampling."""
    ny, nx = shape
    x0 = max(0, int(math.floor(cx - radius_outer)))
    x1 = min(nx - 1, int(math.ceil(cx + radius_outer)))
    y0 = max(0, int(math.floor(cy - radius_outer)))
    y1 = min(ny - 1, int(math.ceil(cy + radius_outer)))
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    if width <= 0 or height <= 0:
        empty = np.zeros((0, 0), dtype=np.float64)
        return empty, (slice(0, 0), slice(0, 0))

    offsets = (np.arange(oversample) + 0.5) / oversample - 0.5
    xs = np.arange(x0, x1 + 1, dtype=np.float64)
    ys = np.arange(y0, y1 + 1, dtype=np.float64)
    xx = (xs[:, None, None, None] + offsets[None, None, None, :]) - cx
    yy = (ys[None, :, None, None] + offsets[None, None, :, None]) - cy
    distance = xx ** 2 + yy ** 2  # (width, height, oversample, oversample)
    inside = (distance >= radius_inner ** 2) & (distance < radius_outer ** 2)
    coverage = inside.mean(axis=(2, 3)).T
    return coverage, (slice(y0, y1 + 1), slice(x0, x1 + 1))


def _overall_quality(
    detections: list[DetectedSource], measurements: list[FluxMeasurement]
) -> SourceQuality:
    if not detections or not measurements:
        return "WARNING"
    qualities = {item.quality for item in measurements}
    if "FAIL" in qualities:
        return "FAIL"
    if "NOT_VERIFIED" in qualities:
        return "NOT_VERIFIED"
    if "WARNING" in qualities:
        return "WARNING"
    return "PASS"


def _cache_key(response: DetectionResponse, config: PhotometryConfig) -> str:
    payload = {
        "detection": response.provenance.cache_key,
        "config": config.model_dump(),
        "algorithm": PHOTOMETRY_ALGORITHM_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _round(value: float | None) -> float | None:
    return round(value, 12) if value is not None and math.isfinite(value) else None


__all__ = ["PhotometryService", "measure_prepared"]
