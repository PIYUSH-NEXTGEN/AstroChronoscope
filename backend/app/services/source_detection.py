"""Phase 3 source detection.

Baseline algorithm (documented, testable, no hidden library behaviour):

1. Prepare the image (valid mask, background, noise) via ``image_preparation``.
2. Smooth the background-subtracted image with a Gaussian matched filter while
   normalising by the smoothed validity mask, so masked pixels never create
   false peaks.
3. Threshold at ``threshold_sigma`` times the propagated smoothed noise.
4. Label 8-connected components; optionally split a component that contains
   several local maxima (simple peak-assignment deblending, not a full
   multi-source model).
5. Measure flux-weighted centroids, second-moment shapes and integrated SNR.
6. Convert centroids to ICRS with the product WCS.

This is a point-source/compact-source baseline. Extended emission, severe
crowding and correlated noise are flagged or documented limitations, not
silently corrected. A ``detection_snr`` is a pixel-sum significance, never a
calibrated flux.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, label as nd_label, maximum_filter

from app.core.cache import CacheStore
from app.core.exceptions import InvalidCoordinateError
from app.core.versions import software_versions
from app.schemas.common import Target
from app.schemas.sources import (
    DETECTION_ALGORITHM_VERSION,
    DetectedSource,
    DetectionConfig,
    DetectionProvenance,
    DetectionQuality,
    DetectionRequest,
    DetectionResponse,
    SourceQuality,
)
from app.services.artifact_resolver import resolve_observation_artifact
from app.services.image_inspection import file_sha256
from app.services.image_preparation import PreparedImage, prepare_image
from app.services.observation_service import ObservationService

log = logging.getLogger(__name__)

_FWHM_PER_SIGMA = 2.3548200450309493
_STRUCTURE = np.ones((3, 3), dtype=bool)
#: A second local maximum is only treated as a blend if it reaches this fraction
#: of the component's brightest peak. Noise bumps on a bright source's wings
#: otherwise masquerade as independent peaks.
_DEBLEND_MIN_PEAK_FRACTION = 0.25
_SEMAPHORE: threading.BoundedSemaphore | None = None
_SEMAPHORE_LOCK = threading.Lock()


def _semaphore(limit: int) -> threading.BoundedSemaphore:
    global _SEMAPHORE
    with _SEMAPHORE_LOCK:
        if _SEMAPHORE is None:
            _SEMAPHORE = threading.BoundedSemaphore(max(1, limit))
        return _SEMAPHORE


@dataclass
class DetectionResult:
    prepared: PreparedImage
    detections: list[DetectedSource]
    truncated: bool
    quality: DetectionQuality


class SourceDetectionService:
    """Detect compact sources in one resolved SPHEREx product."""

    def __init__(self, observations: ObservationService) -> None:
        self.observations = observations
        self.settings = observations.settings
        self.cache: CacheStore = observations.cache
        self.settings.ensure_dirs()

    def detect(self, request: DetectionRequest) -> DetectionResponse:
        if (request.ra is None) != (request.dec is None):
            raise InvalidCoordinateError(
                "ra and dec must be provided together", details={"ra": request.ra, "dec": request.dec}
            )

        key = _cache_key(request)
        cached = self.cache.sources.get(key)
        if cached is not None:
            return cached

        semaphore = _semaphore(self.settings.max_concurrent_detections)
        if not semaphore.acquire(timeout=self.settings.request_timeout):
            from app.core.exceptions import ProcessingBusyError

            raise ProcessingBusyError(
                "Too many source detections are running; retry shortly",
                details={"max_concurrent": self.settings.max_concurrent_detections},
            )
        size = (
            request.size if request.size is not None else self.settings.comparison_size_deg
        ) if request.ra is not None else None
        try:
            resolved = resolve_observation_artifact(
                self.observations,
                request.observation_id,
                request.ra,
                request.dec,
                size,
                default_size=self.settings.comparison_size_deg,
            )
            prepared = prepare_image(resolved.path, mesh=request.config.background_mesh)
            result = detect_prepared(
                prepared,
                request.config,
                observation_key=resolved.detail.obs_publisher_did or request.observation_id,
            )
        finally:
            semaphore.release()

        response = DetectionResponse(
            observation_id=request.observation_id,
            source=resolved.source,
            source_url=resolved.detail.fits_access_url,
            target=Target(ra=request.ra, dec=request.dec) if request.ra is not None else None,
            size_deg=size,
            config=request.config,
            image=prepared.summary(),
            background=prepared.background,
            detections=result.detections,
            total_detections=len(result.detections),
            truncated=result.truncated,
            quality=result.quality,
            provenance=DetectionProvenance(
                created_at=datetime.now(timezone.utc).isoformat(),
                software=software_versions(),
                algorithm=(
                    "mask-normalised Gaussian matched filter; thresholded connected components; "
                    "peak-assignment deblending; flux-weighted second-moment shapes"
                ),
                observation_id=request.observation_id,
                obs_publisher_did=resolved.detail.obs_publisher_did,
                band=resolved.detail.band,
                observation_time=resolved.detail.start_time,
                fits_sha256=file_sha256(resolved.path),
                cache_key=key,
                config=request.config.model_dump(),
            ),
        )
        self.cache.sources.set(key, response)
        log.info(
            "sources_detected",
            extra={
                "observation_id": request.observation_id,
                "detections": len(result.detections),
                "quality": result.quality.overall_status,
            },
        )
        return response


# ------------------------------------------------------------------ algorithm
def detect_prepared(
    prepared: PreparedImage,
    config: DetectionConfig,
    *,
    observation_key: str = "unknown",
) -> DetectionResult:
    """Run the detection algorithm on an already prepared image."""
    valid_pixels = int(prepared.valid.sum())
    if valid_pixels == 0:
        return DetectionResult(
            prepared=prepared,
            detections=[],
            truncated=False,
            quality=DetectionQuality(
                overall_status="FAIL",
                warnings=["no_valid_pixels"],
            ),
        )

    noise = prepared.noise
    background = prepared.background
    if background.level is None or noise is None or noise <= 0:
        return DetectionResult(
            prepared=prepared,
            detections=[],
            truncated=False,
            quality=DetectionQuality(
                overall_status="NOT_VERIFIED",
                warnings=["background_or_noise_unavailable"],
            ),
        )

    sigma_px = max(config.filter_fwhm_px / _FWHM_PER_SIGMA, 0.1)
    work = prepared.data - float(background.level)
    smoothed, weights, noise_factor = _masked_smooth(work, prepared.valid, sigma_px)
    smooth_noise = float(noise) * noise_factor
    if not math.isfinite(smooth_noise) or smooth_noise <= 0:
        return DetectionResult(
            prepared=prepared,
            detections=[],
            truncated=False,
            quality=DetectionQuality(
                overall_status="NOT_VERIFIED",
                warnings=["smoothed_noise_unavailable"],
            ),
        )

    threshold = config.threshold_sigma * smooth_noise
    usable = prepared.valid & np.isfinite(smoothed) & (weights >= 0.5)
    detection_map = usable & (smoothed > threshold)
    labels, count = nd_label(detection_map, structure=_STRUCTURE)

    maxima = usable & (smoothed == maximum_filter(np.where(usable, smoothed, -np.inf), size=3, mode="constant", cval=-np.inf))
    max_labels, _ = nd_label(maxima, structure=_STRUCTURE)
    invalid_dilated = (
        binary_dilation(~prepared.valid, structure=_STRUCTURE) if not prepared.valid.all() else np.zeros_like(prepared.valid)
    )

    detections: list[DetectedSource] = []
    for component in range(1, count + 1):
        component_pixels = labels == component
        area = int(component_pixels.sum())
        if area < config.min_area_px:
            continue

        ys, xs = np.nonzero(component_pixels)
        peaks = _component_peaks(component_pixels, maxima, max_labels, smoothed)
        if config.deblend and len(peaks) > 1:
            groups = _assign_to_peaks(ys, xs, peaks)
        else:
            groups = [(ys, xs, peaks[0] if peaks else _peak_from_pixels(ys, xs, smoothed))]

        for group_ys, group_xs, _peak in groups:
            detection = _measure_group(
                group_ys,
                group_xs,
                prepared,
                work,
                noise,
                config,
                observation_key,
                invalid_dilated,
                blended=len(peaks) > 1,
            )
            if detection is not None:
                detections.append(detection)

    detections.sort(
        key=lambda item: (
            -(item.detection_snr if item.detection_snr is not None else -math.inf),
            -(item.peak_value if item.peak_value is not None else -math.inf),
            item.detection_id,
        )
    )
    truncated = len(detections) > config.max_sources
    if truncated:
        detections = detections[: config.max_sources]

    warnings = _aggregate_warnings(prepared, detections, truncated)
    if not detections and "no_sources_detected" not in warnings:
        warnings.append("no_sources_detected")
    overall: SourceQuality = "WARNING" if warnings else "PASS"
    return DetectionResult(
        prepared=prepared,
        detections=detections,
        truncated=truncated,
        quality=DetectionQuality(overall_status=overall, warnings=sorted(warnings)),
    )


def detect_fits(
    fits_path: Path | str,
    config: DetectionConfig | None = None,
    *,
    observation_key: str = "unknown",
    mesh: int = 4,
) -> DetectionResult:
    """Convenience wrapper used by tests and scripts."""
    prepared = prepare_image(fits_path, mesh=(config.background_mesh if config else mesh))
    return detect_prepared(prepared, config or DetectionConfig(), observation_key=observation_key)


# -------------------------------------------------------------------- helpers
def _masked_smooth(values: np.ndarray, valid: np.ndarray, sigma_px: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Gaussian smooth that ignores invalid pixels and returns the noise factor."""
    mask = valid.astype(np.float64)
    weights = gaussian_filter(mask, sigma=sigma_px, mode="constant", truncate=4.0)
    numerator = gaussian_filter(np.where(valid, values, 0.0), sigma=sigma_px, mode="constant", truncate=4.0)
    smoothed = np.full(values.shape, np.nan, dtype=np.float64)
    ok = weights > 0
    smoothed[ok] = numerator[ok] / weights[ok]
    return smoothed, weights, _kernel_noise_factor(sigma_px)


def _kernel_noise_factor(sigma_px: float) -> float:
    """sqrt(sum(k^2)) for the Gaussian kernel actually used by the filter."""
    radius = max(1, int(math.ceil(4.0 * sigma_px)))
    size = 2 * radius + 1
    impulse = np.zeros((size, size), dtype=np.float64)
    impulse[radius, radius] = 1.0
    kernel = gaussian_filter(impulse, sigma=sigma_px, mode="constant", truncate=4.0)
    return float(np.sqrt(np.sum(kernel ** 2)))


def _component_peaks(
    component_pixels: np.ndarray,
    maxima: np.ndarray,
    max_labels: np.ndarray,
    smoothed: np.ndarray,
) -> list[tuple[float, float, float]]:
    peak_ids = np.unique(max_labels[component_pixels & maxima])
    peaks: list[tuple[float, float, float]] = []
    for peak_id in peak_ids:
        if peak_id == 0:
            continue
        ys, xs = np.nonzero((max_labels == peak_id) & maxima)
        if not ys.size:
            continue
        values = smoothed[ys, xs]
        index = int(np.nanargmax(values))
        peaks.append((float(xs[index]), float(ys[index]), float(values[index])))
    if not peaks:
        ys, xs = np.nonzero(component_pixels)
        peaks.append(_peak_from_pixels(ys, xs, smoothed))
    peaks.sort(key=lambda item: item[2], reverse=True)
    if len(peaks) > 1:
        brightest = peaks[0][2]
        peaks = [peak for peak in peaks if peak[2] >= _DEBLEND_MIN_PEAK_FRACTION * brightest]
    return peaks


def _peak_from_pixels(ys: np.ndarray, xs: np.ndarray, smoothed: np.ndarray) -> tuple[float, float, float]:
    values = smoothed[ys, xs]
    index = int(np.nanargmax(values))
    return float(xs[index]), float(ys[index]), float(values[index])


def _assign_to_peaks(
    ys: np.ndarray, xs: np.ndarray, peaks: list[tuple[float, float, float]]
) -> list[tuple[np.ndarray, np.ndarray, tuple[float, float, float]]]:
    peak_array = np.asarray([(px, py) for px, py, _ in peaks], dtype=np.float64)
    distances = (xs[:, None] - peak_array[None, :, 0]) ** 2 + (ys[:, None] - peak_array[None, :, 1]) ** 2
    assignment = np.argmin(distances, axis=1)
    groups: list[tuple[np.ndarray, np.ndarray, tuple[float, float, float]]] = []
    for index, peak in enumerate(peaks):
        selected = assignment == index
        if selected.any():
            groups.append((ys[selected], xs[selected], peak))
    return groups or [(ys, xs, peaks[0])]


def _measure_group(
    ys: np.ndarray,
    xs: np.ndarray,
    prepared: PreparedImage,
    work: np.ndarray,
    noise: float,
    config: DetectionConfig,
    observation_key: str,
    invalid_dilated: np.ndarray,
    *,
    blended: bool,
) -> DetectedSource | None:
    area = int(ys.size)
    if area < config.min_area_px:
        return None

    flux = float(np.sum(work[ys, xs]))
    if not math.isfinite(flux) or flux <= 0:
        return None

    raw = np.maximum(work[ys, xs], 0.0)
    weight_sum = float(raw.sum())
    if weight_sum <= 0:
        return None
    cx = float(np.sum(raw * xs) / weight_sum)
    cy = float(np.sum(raw * ys) / weight_sum)

    dx = xs - cx
    dy = ys - cy
    mxx = float(np.sum(raw * dx * dx) / weight_sum)
    myy = float(np.sum(raw * dy * dy) / weight_sum)
    mxy = float(np.sum(raw * dx * dy) / weight_sum)
    shape = _shape(mxx, myy, mxy)

    snr = flux / (noise * math.sqrt(area)) if noise > 0 else None
    fwhm_px = shape["fwhm_px"]
    centroid_error_px = (
        fwhm_px / (_FWHM_PER_SIGMA * snr)
        if fwhm_px is not None and snr is not None and snr > 0
        else None
    )
    peak_value = float(np.max(prepared.data[ys, xs]))
    ny, nx = prepared.shape
    margin = config.edge_margin_px
    touches_edge = bool(
        xs.min() <= 0 or ys.min() <= 0 or xs.max() >= nx - 1 or ys.max() >= ny - 1
    )
    edge = touches_edge or cx < margin or cx > (nx - 1 - margin) or cy < margin or cy > (ny - 1 - margin)
    near_invalid = bool(invalid_dilated[ys, xs].any())

    ra, dec, wcs_ok = _world_coordinates(prepared, cx, cy)
    flags: list[str] = []
    if blended:
        flags.append("blended")
    if edge:
        flags.append("edge")
    if near_invalid:
        flags.append("near_invalid")
    if not wcs_ok:
        flags.append("wcs_unavailable")
    if snr is None or snr < config.threshold_sigma:
        flags.append("low_snr")
    if shape["fwhm_px"] is None:
        flags.append("shape_unresolved")

    quality: SourceQuality = "PASS" if not flags else "WARNING"
    detection_id = hashlib.sha256(f"{observation_key}|{cx:.4f}|{cy:.4f}".encode("utf-8")).hexdigest()[:32]
    return DetectedSource(
        detection_id=detection_id,
        source_id=None,
        x=round(cx, 4),
        y=round(cy, 4),
        ra_deg=ra,
        dec_deg=dec,
        peak_value=_finite(peak_value),
        source_area=area,
        centroid_x=round(cx, 4),
        centroid_y=round(cy, 4),
        detection_snr=_finite(snr),
        centroid_error_px=_finite(centroid_error_px),
        centroid_error_arcsec=(
            centroid_error_px * prepared.pixel_scale_arcsec
            if centroid_error_px is not None and prepared.pixel_scale_arcsec is not None
            else None
        ),
        fwhm_px=fwhm_px,
        fwhm_arcsec=(
            fwhm_px * prepared.pixel_scale_arcsec
            if fwhm_px is not None and prepared.pixel_scale_arcsec is not None
            else None
        ),
        ellipticity=shape["ellipticity"],
        orientation_deg=shape["orientation_deg"],
        background_level=prepared.background.level,
        noise_sigma=_finite(noise),
        blended=blended,
        flags=flags,
        quality=quality,
    )


def _shape(mxx: float, myy: float, mxy: float) -> dict[str, float | None]:
    trace = mxx + myy
    determinant = mxx * myy - mxy * mxy
    discriminant = max(trace * trace / 4.0 - determinant, 0.0)
    root = math.sqrt(discriminant)
    major = max(trace / 2.0 + root, 0.0)
    minor = max(trace / 2.0 - root, 0.0)
    sigma_major = math.sqrt(major)
    sigma_minor = math.sqrt(minor)
    fwhm = _FWHM_PER_SIGMA * sigma_major if sigma_major > 0 else None
    ellipticity = (1.0 - sigma_minor / sigma_major) if sigma_major > 0 else None
    orientation = math.degrees(0.5 * math.atan2(2.0 * mxy, mxx - myy))
    return {
        "fwhm_px": _finite(fwhm),
        "ellipticity": _finite(ellipticity),
        "orientation_deg": _finite(orientation),
    }


def _world_coordinates(prepared: PreparedImage, x: float, y: float) -> tuple[float | None, float | None, bool]:
    if prepared.wcs is None:
        return None, None, False
    try:
        world = prepared.wcs.pixel_to_world(x, y)
        return _finite(float(world.ra.deg)), _finite(float(world.dec.deg)), True
    except Exception:  # pragma: no cover - defensive
        return None, None, False


def _aggregate_warnings(
    prepared: PreparedImage, detections: list[DetectedSource], truncated: bool
) -> list[str]:
    warnings: set[str] = set()
    if not prepared.background.reliable:
        warnings.add("background_unreliable")
    if prepared.variance is None:
        warnings.add("no_variance_extension")
    if prepared.wcs is None:
        warnings.add("missing_wcs")
    if prepared.psf_fwhm_arcsec is None:
        warnings.add("psf_unavailable")
    if truncated:
        warnings.add("detection_limit_applied")
    if any("blended" in item.flags for item in detections):
        warnings.add("blended_sources_present")
    if any("edge" in item.flags for item in detections):
        warnings.add("edge_sources_present")
    if any("near_invalid" in item.flags for item in detections):
        warnings.add("sources_near_invalid_pixels")
    if not detections:
        warnings.add("no_sources_detected")
    return sorted(warnings)


def _cache_key(request: DetectionRequest) -> str:
    payload = {
        "observation_id": request.observation_id,
        "ra": round(request.ra, 8) if request.ra is not None else None,
        "dec": round(request.dec, 8) if request.dec is not None else None,
        "size": round(request.size, 6) if request.size is not None else None,
        "config": request.config.model_dump(),
        "algorithm": DETECTION_ALGORITHM_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


__all__ = [
    "DetectionResult",
    "SourceDetectionService",
    "detect_fits",
    "detect_prepared",
]
