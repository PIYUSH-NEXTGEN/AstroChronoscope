"""Phase 3 image preparation: validity mask, variance, background and noise.

Everything here is derived from the FITS product itself. Missing information is
reported as missing; invalid pixels are never replaced by zero.

Pixel solid angle
-----------------
SPHEREx level-2 ``IMAGE`` pixels store surface brightness (MJy/sr). Converting an
aperture sum to a physical flux requires the solid angle each pixel subtends.
This module reports a constant per-pixel solid angle from the projection-plane
pixel scale (``projection_plane_constant``). For the small SPHEREx cutouts used
here the field distortion is negligible; a full distortion-aware map is not
claimed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS

from app.schemas.sources import BackgroundEstimate, ImagePreparationSummary
from app.services.fits_service import ScienceImage, extract_science_image
from app.services.image_inspection import invalid_flag_mask, pixel_scale_arcsec, psf_summary
from app.services.wcs_service import celestial_wcs

_MIN_BACKGROUND_PIXELS = 25


@dataclass
class PreparedImage:
    """A validated science image plus the products later phases need."""

    data: np.ndarray
    valid: np.ndarray
    variance: np.ndarray | None
    flags: np.ndarray | None
    flag_masks: dict[str, int] | None
    header: fits.Header
    wcs: WCS | None
    science_extname: str
    bunit: str | None
    available_hdus: list[str]
    background: BackgroundEstimate
    noise: float | None
    pixel_scale_arcsec: float | None
    pixel_solid_angle_sr: float | None
    psf_fwhm_arcsec: float | None
    nan_pixels: int
    inf_pixels: int
    masked_pixels: int
    valid_fraction: float
    variance_valid_fraction: float | None

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.data.shape[0]), int(self.data.shape[1])

    @property
    def mask_bits(self) -> dict[str, int]:
        return dict(self.flag_masks or {})

    def summary(self) -> ImagePreparationSummary:
        ny, nx = self.shape
        invalid = int(self.valid.size - self.valid.sum())
        return ImagePreparationSummary(
            shape=[ny, nx],
            science_extname=self.science_extname,
            bunit=self.bunit,
            valid_pixels=int(self.valid.sum()),
            invalid_pixels=invalid,
            masked_pixels=self.masked_pixels,
            nan_pixels=self.nan_pixels,
            inf_pixels=self.inf_pixels,
            valid_fraction=self.valid_fraction,
            variance_available=self.variance is not None,
            variance_valid_fraction=self.variance_valid_fraction,
            pixel_scale_arcsec=self.pixel_scale_arcsec,
            pixel_solid_angle_sr=self.pixel_solid_angle_sr,
            pixel_solid_angle_model="projection_plane_constant" if self.pixel_solid_angle_sr else "unavailable",
            psf_fwhm_arcsec=self.psf_fwhm_arcsec,
            mask_bits=self.mask_bits,
        )


def prepare_image(fits_path: Path | str, *, mesh: int = 4) -> PreparedImage:
    """Extract, mask and characterise one science image.

    The returned arrays are copies; the FITS file itself is never modified.
    """
    path = Path(fits_path)
    science = extract_science_image(path)
    wcs = celestial_wcs(science.header)
    scale = pixel_scale_arcsec(wcs) if wcs is not None else None
    solid_angle = _pixel_solid_angle_sr(wcs)

    finite = np.isfinite(science.data)
    masked = np.zeros_like(finite)
    if science.flags is not None:
        masked = invalid_flag_mask(science.flags, science.flag_masks)
    valid = finite & ~masked

    variance = _valid_variance(science, valid)
    variance_valid_fraction = None
    if variance is not None:
        count = int(valid.sum())
        variance_valid_fraction = float(np.isfinite(variance[valid]).sum() / count) if count else 0.0

    nan_pixels = int(np.isnan(science.data).sum())
    inf_pixels = int(np.isinf(science.data).sum())
    masked_pixels = int(np.count_nonzero(finite & masked))
    total = int(science.data.size)
    valid_fraction = float(valid.sum() / total) if total else 0.0

    background = estimate_background(science.data, valid, mesh=mesh)
    noise = background.sigma if background.reliable else None
    psf = psf_summary(path, scale)

    return PreparedImage(
        data=np.array(science.data, dtype=np.float64, copy=True),
        valid=valid,
        variance=variance,
        flags=science.flags,
        flag_masks=science.flag_masks,
        header=science.header,
        wcs=wcs,
        science_extname=science.extname,
        bunit=str(science.unit) if science.unit else None,
        available_hdus=list(science.available_hdus),
        background=background,
        noise=noise,
        pixel_scale_arcsec=scale,
        pixel_solid_angle_sr=solid_angle,
        psf_fwhm_arcsec=psf.fwhm_arcsec,
        nan_pixels=nan_pixels,
        inf_pixels=inf_pixels,
        masked_pixels=masked_pixels,
        valid_fraction=valid_fraction,
        variance_valid_fraction=variance_valid_fraction,
    )


def estimate_background(data: np.ndarray, valid: np.ndarray, *, mesh: int = 4) -> BackgroundEstimate:
    """Sigma-clipped global median, its clipped sigma, and a mesh variation."""
    values = data[valid]
    if values.size < _MIN_BACKGROUND_PIXELS:
        return BackgroundEstimate(
            method="none",
            valid_pixels=int(values.size),
            reliable=False,
            note=f"Fewer than {_MIN_BACKGROUND_PIXELS} valid pixels; background and noise are unavailable",
        )

    _, level, sigma = sigma_clipped_stats(values, sigma=3.0, maxiters=5)
    level = float(level)
    sigma = float(sigma)

    medians: list[float] = []
    ny, nx = data.shape
    for rows in np.array_split(np.arange(ny), max(1, mesh)):
        for columns in np.array_split(np.arange(nx), max(1, mesh)):
            block = data[np.ix_(rows, columns)]
            block_valid = valid[np.ix_(rows, columns)]
            if block_valid.sum() < max(_MIN_BACKGROUND_PIXELS, int(0.2 * block.size)):
                continue
            _, block_median, _ = sigma_clipped_stats(block[block_valid], sigma=3.0, maxiters=5)
            if math.isfinite(block_median):
                medians.append(float(block_median))
    variation = float(np.std(medians)) if len(medians) >= 2 else None
    tolerance = max(5.0 * sigma, 0.5 * abs(level), 1e-30)
    reliable = math.isfinite(level) and math.isfinite(sigma) and sigma > 0 and (variation is None or variation <= tolerance)

    return BackgroundEstimate(
        method="sigma_clipped_median",
        level=_round(level),
        sigma=_round(sigma),
        mesh_variation=_round(variation),
        reliable=bool(reliable),
        valid_pixels=int(values.size),
        note=(
            f"Sigma-clipped median over {int(values.size)} valid pixels "
            f"({len(medians)} mesh blocks); reliable={bool(reliable)}"
        ),
    )


def _valid_variance(science: ScienceImage, valid: np.ndarray) -> np.ndarray | None:
    if science.variance is None:
        return None
    variance = np.array(science.variance, dtype=np.float64, copy=True)
    bad = ~np.isfinite(variance) | (variance < 0.0)
    variance[bad] = np.nan
    variance[~valid] = np.nan
    return variance


def _pixel_solid_angle_sr(wcs: WCS | None) -> float | None:
    if wcs is None:
        return None
    try:
        scales = wcs.proj_plane_pixel_scales()
    except Exception:  # pragma: no cover - defensive
        return None
    try:
        x = abs(float(scales[0].to_value(u.deg)))
        y = abs(float(scales[1].to_value(u.deg)))
    except Exception:  # pragma: no cover - defensive
        return None
    area = x * y
    if not math.isfinite(area) or area <= 0:
        return None
    return area * (math.pi / 180.0) ** 2


def _round(value: float | None) -> float | None:
    return round(value, 12) if value is not None and math.isfinite(value) else None


__all__ = ["PreparedImage", "estimate_background", "prepare_image"]
