"""FITS image inspection: HDUs, WCS, masks, uncertainty, PSF and statistics.

Everything reported here comes from the file itself or from the SIA2 row that
resolved it. Missing information is reported as missing, never inferred.
"""

from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS

from app.schemas.image_processing import (
    HduInfo,
    ImageHeaderMetadata,
    ImageInspection,
    ImageStatistics,
    InspectionWarning,
    MaskInfo,
    PsfInfo,
    VarianceInfo,
    WcsValidation,
)
from app.services.fits_service import ScienceImage, extract_science_image, verify_fits
from app.services.wcs_service import celestial_wcs, validate_target_in_cutout

log = logging.getLogger(__name__)

_WARNING_MESSAGES: dict[str, str] = {
    "missing_wcs": "FITS header has no usable celestial WCS",
    "all_pixels_invalid": "Science image contains no finite pixels",
    "non_finite_pixels": "Science image contains NaN and/or infinite pixels",
    "no_flags_extension": "No FLAGS mask extension found; only non-finite pixels are masked",
    "no_variance_extension": "No VARIANCE extension found; uncertainty propagation is unavailable",
    "no_psf_extension": "No EPSF extension found; PSF compatibility cannot be characterised",
    "psf_scale_unknown": "EPSF present but the oversampling factor is missing; FWHM not converted to arcsec",
    "target_outside_image": "Target position falls outside this image footprint",
    "no_bunit": "Science image has no BUNIT; photometric units are unknown",
}

# FLAGS bits that annotate a pixel rather than mark it unusable. Names come from
# the product's own MSKN/MSKD definitions: SOURCE = "Pixel mapped to a known
# source", FULLSAMPLE = "Pixel full sample history is available". Masking them
# would discard every real source. Unknown bits stay invalidating (conservative).
INFORMATIONAL_FLAG_BITS = frozenset({"SOURCE", "FULLSAMPLE"})
_MASK_POLICY = (
    "any non-zero FLAGS bit marks the pixel invalid, except informational bits "
    "(SOURCE, FULLSAMPLE)"
)


def invalid_flag_mask(flags: np.ndarray, bit_masks: dict[str, int] | None) -> np.ndarray:
    """Pixels whose FLAGS carry a non-informational bit."""
    informational = 0
    for name, value in (bit_masks or {}).items():
        if name.upper() in INFORMATIONAL_FLAG_BITS:
            informational |= int(value)
    return (np.asarray(flags, dtype=np.int64) & ~informational) != 0


def inspect_fits(
    path: Path | str,
    *,
    observation_id: str | None = None,
    obs_publisher_did: str | None = None,
    band: str | None = None,
    target_ra: float | None = None,
    target_dec: float | None = None,
    source: str = "cutout",
    source_url: str = "",
) -> ImageInspection:
    """Validate and summarise one FITS product."""
    path = Path(path)
    verify_fits(path)
    image = extract_science_image(path)

    wcs_validation = _wcs_validation(image.header, image.data.shape, target_ra, target_dec)
    dtype, mask_bits, psf, variance_unit = _aux_metadata(
        path, image.extname, image.data.shape, wcs_validation.pixel_scale_arcsec
    )
    stats = _statistics(image, dtype)
    mask = _mask_info(image, mask_bits)
    variance = _variance_info(image, variance_unit)

    warnings: list[str] = []
    if not wcs_validation.available:
        warnings.append("missing_wcs")
    if stats.finite_pixels == 0:
        warnings.append("all_pixels_invalid")
    elif stats.nan_pixels or stats.inf_pixels:
        warnings.append("non_finite_pixels")
    if not mask.available:
        warnings.append("no_flags_extension")
    if not variance.available:
        warnings.append("no_variance_extension")
    if not psf.available:
        warnings.append("no_psf_extension")
    elif psf.fwhm_arcsec is None:
        warnings.append("psf_scale_unknown")
    if wcs_validation.target_inside_image is False:
        warnings.append("target_outside_image")
    if not stats.bunit:
        warnings.append("no_bunit")

    if stats.finite_pixels == 0:
        quality = "FAIL"
    elif warnings:
        quality = "WARNING"
    else:
        quality = "PASS"

    return ImageInspection(
        source="product" if source == "product" else "cutout",
        source_url=source_url,
        fits_sha256=file_sha256(path),
        science_extname=image.extname,
        hdus=_hdu_list(path),
        stats=stats,
        wcs=wcs_validation,
        mask=mask,
        variance=variance,
        psf=psf,
        metadata=_header_metadata(image, observation_id, obs_publisher_did, band),
        quality=quality,
        warnings=[InspectionWarning(code=code, message=_WARNING_MESSAGES.get(code, code)) for code in warnings],
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------- sections
def _hdu_list(path: Path) -> list[HduInfo]:
    result: list[HduInfo] = []
    with fits.open(path, memmap=True) as hdul:
        for index, hdu in enumerate(hdul):
            data = hdu.data
            if index == 0:
                kind = "primary"
            elif isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                kind = "table"
            elif data is not None:
                kind = "image"
            else:
                kind = "primary"
            result.append(
                HduInfo(
                    index=index,
                    name=str(hdu.name).upper(),
                    kind=kind,
                    shape=list(data.shape) if getattr(data, "ndim", 0) >= 2 else None,
                    dtype=str(data.dtype) if hasattr(data, "dtype") else None,
                    unit=_text(hdu.header.get("BUNIT")),
                )
            )
    return result


def _statistics(image: ScienceImage, dtype: str) -> ImageStatistics:
    data = image.data
    finite = np.isfinite(data)
    nan_pixels = int(np.isnan(data).sum())
    inf_pixels = int(np.isinf(data).sum())
    total = int(data.size)
    finite_pixels = int(finite.sum())

    valid = finite.copy()
    if image.flags is not None:
        valid &= ~invalid_flag_mask(image.flags, image.flag_masks)
    valid_pixels = int(valid.sum())

    values = data[finite]
    return ImageStatistics(
        shape=[int(data.shape[0]), int(data.shape[1])],
        dtype=dtype,
        bunit=_text(image.unit),
        finite_pixels=finite_pixels,
        nan_pixels=nan_pixels,
        inf_pixels=inf_pixels,
        finite_fraction=finite_pixels / total if total else 0.0,
        valid_pixels=valid_pixels,
        valid_fraction=valid_pixels / total if total else 0.0,
        min=float(values.min()) if values.size else None,
        max=float(values.max()) if values.size else None,
        mean=float(values.mean()) if values.size else None,
        median=float(np.median(values)) if values.size else None,
        std=float(values.std()) if values.size else None,
    )


def _wcs_validation(
    header: fits.Header,
    shape: tuple[int, ...],
    target_ra: float | None,
    target_dec: float | None,
) -> WcsValidation:
    wcs = celestial_wcs(header)
    if wcs is None:
        return WcsValidation(available=False, message="FITS header has no celestial WCS")

    ctype = [str(value) for value in (wcs.wcs.ctype or [])]
    pixel_scale = pixel_scale_arcsec(wcs)
    ny, nx = int(shape[0]), int(shape[1])

    footprint: list[list[float]] | None = None
    try:
        footprint = [[float(ra), float(dec)] for ra, dec in wcs.calc_footprint()]
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("wcs_footprint_failed", extra={"error": str(exc)})

    center = _world_to_radec(wcs, (nx - 1) / 2.0, (ny - 1) / 2.0)
    target_inside: bool | None = None
    target_pixel: list[float] | None = None
    message: str | None = None
    if target_ra is not None and target_dec is not None:
        info = validate_target_in_cutout(target_ra, target_dec, wcs, shape)
        target_inside = info.target_inside_image
        if info.pixel_x is not None and info.pixel_y is not None:
            target_pixel = [info.pixel_x, info.pixel_y]
        message = info.message

    return WcsValidation(
        available=True,
        ctype=ctype,
        projection=_projection(ctype),
        has_sip=wcs.sip is not None,
        crval_ra_deg=_float(header.get("CRVAL1")),
        crval_dec_deg=_float(header.get("CRVAL2")),
        crpix=_pair(header.get("CRPIX1"), header.get("CRPIX2")),
        pixel_scale_arcsec=pixel_scale,
        orientation_deg=_orientation_deg(wcs),
        footprint=footprint,
        center_ra_deg=center[0],
        center_dec_deg=center[1],
        target_inside_image=target_inside,
        target_pixel=target_pixel,
        message=message,
    )


def _aux_metadata(
    path: Path, extname: str, shape: tuple[int, ...], pixel_scale_arcsec: float | None
) -> tuple[str, dict[str, str], PsfInfo, str | None]:
    dtype = "unknown"
    bits: dict[str, str] = {}
    variance_unit: str | None = None
    psf = PsfInfo(available=False, message="No EPSF extension")
    with fits.open(path, memmap=True) as hdul:
        for hdu in hdul:
            name = str(hdu.name).upper()
            if name == extname and hdu.data is not None:
                dtype = str(hdu.data.dtype)
            elif name == "FLAGS" and hdu.data is not None:
                bits = _mask_bits(hdu.header)
            elif name == "VARIANCE" and hdu.data is not None:
                variance_unit = _text(hdu.header.get("BUNIT"))
            elif name == "EPSF" and hdu.data is not None:
                psf = _psf_info(hdu, pixel_scale_arcsec)
    return dtype, bits, psf, variance_unit


def _mask_bits(header: fits.Header) -> dict[str, str]:
    bits: dict[str, str] = {}
    for index in range(0, 32):
        name = header.get(f"MSKN{index:04d}")
        if name:
            bits[str(name)] = str(header.get(f"MSKD{index:04d}") or "")
    return bits


def _psf_info(hdu: fits.BinTableHDU, pixel_scale_arcsec: float | None) -> PsfInfo:
    table = hdu.data
    names = table.dtype.names or ()
    if "EPSF" not in names:
        return PsfInfo(available=False, message="EPSF extension has no EPSF column")

    kernels = np.asarray(table["EPSF"], dtype=float)
    if kernels.ndim != 3 or not kernels.size:
        return PsfInfo(available=False, message="EPSF column is not a stack of 2-D kernels")

    median_kernel = np.median(kernels, axis=0)
    cy, cx = np.unravel_index(int(np.argmax(median_kernel)), median_kernel.shape)
    row_width = _half_max_width(median_kernel[cy, :], int(cx))
    column_width = _half_max_width(median_kernel[:, cx], int(cy))
    fwhm_samples = None
    if row_width is not None and column_width is not None:
        fwhm_samples = (row_width + column_width) / 2.0

    oversampling = _int(hdu.header.get("OVSMPX")) or _int(hdu.header.get("OVSMPY"))
    fwhm_px = None
    fwhm_arcsec = None
    message = None
    if fwhm_samples is not None and oversampling:
        fwhm_px = fwhm_samples / float(oversampling)
        if pixel_scale_arcsec is not None:
            fwhm_arcsec = fwhm_px * pixel_scale_arcsec
    elif fwhm_samples is not None:
        message = "EPSF oversampling factor missing; FWHM not converted to pixels/arcsec"

    wavelengths = np.asarray(table["CWAVE"], dtype=float) if "CWAVE" in names else None
    stars = np.asarray(table["NSTAR"], dtype=float) if "NSTAR" in names else None
    return PsfInfo(
        available=True,
        extname="EPSF",
        entries=int(kernels.shape[0]),
        oversampling=oversampling,
        fwhm_samples=_round(fwhm_samples, 4),
        fwhm_px=_round(fwhm_px, 4),
        fwhm_arcsec=_round(fwhm_arcsec, 4),
        wavelength_min_um=_round(float(np.nanmin(wavelengths)), 4) if wavelengths is not None and wavelengths.size else None,
        wavelength_max_um=_round(float(np.nanmax(wavelengths)), 4) if wavelengths is not None and wavelengths.size else None,
        stars_median=int(np.nanmedian(stars)) if stars is not None and stars.size else None,
        message=message,
    )


def psf_summary(path: Path | str, pixel_scale_arcsec: float | None) -> PsfInfo:
    """Read the EPSF extension of a product and summarise its median FWHM.

    This is the same measurement ``inspect_fits`` reports, exposed for services
    that need only the PSF and should not pay for a full inspection.
    """
    try:
        with fits.open(Path(path), memmap=True) as hdul:
            for hdu in hdul:
                if str(hdu.name).upper() == "EPSF" and hdu.data is not None:
                    return _psf_info(hdu, pixel_scale_arcsec)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("psf_read_failed", extra={"error": str(exc)})
    return PsfInfo(available=False, message="No EPSF extension")


def _half_max_width(profile: np.ndarray, center: int) -> float | None:
    """Width at half maximum of a 1-D cut through the PSF centre, in samples."""
    if not (0 <= center < profile.size):
        return None
    peak = float(profile[center])
    if not math.isfinite(peak) or peak <= 0:
        return None
    half = peak / 2.0

    def cross(direction: int) -> float | None:
        index = center
        while 0 <= index + direction < profile.size:
            nxt = index + direction
            if profile[nxt] <= half:
                high, low = float(profile[index]), float(profile[nxt])
                fraction = 0.0 if high == low else (high - half) / (high - low)
                return abs(index - center) + fraction
            index = nxt
        return None

    left, right = cross(-1), cross(1)
    if left is None or right is None:
        return None
    return left + right


def _mask_info(image: ScienceImage, bits: dict[str, str]) -> MaskInfo:
    flags = image.flags
    if flags is None:
        return MaskInfo(available=False, policy=None, bits=bits)
    invalid = int(np.count_nonzero(invalid_flag_mask(flags, image.flag_masks)))
    total = int(flags.size)
    return MaskInfo(
        available=True,
        extname="FLAGS",
        invalid_pixels=invalid,
        invalid_fraction=invalid / total if total else 0.0,
        policy=_MASK_POLICY,
        bits=bits,
    )


def _variance_info(image: ScienceImage, unit: str | None) -> VarianceInfo:
    variance = image.variance
    if variance is None:
        return VarianceInfo(available=False)
    finite = np.isfinite(variance)
    values = variance[finite]
    return VarianceInfo(
        available=True,
        extname="VARIANCE",
        unit=unit,
        finite_pixels=int(finite.sum()),
        min=float(values.min()) if values.size else None,
        max=float(values.max()) if values.size else None,
    )


def _header_metadata(
    image: ScienceImage,
    observation_id: str | None,
    obs_publisher_did: str | None,
    band: str | None,
) -> ImageHeaderMetadata:
    header = image.header
    return ImageHeaderMetadata(
        observation_id=observation_id,
        obs_publisher_did=obs_publisher_did,
        obs_id=_text(header.get("OBSID")),
        band=band,
        date_obs=_text(header.get("DATE-OBS")),
        mjd_obs=_float(header.get("MJD-OBS")),
        date_avg=_text(header.get("DATE-AVG")),
        mjd_avg=_float(header.get("MJD-AVG")),
        timesys=_text(header.get("TIMESYS")),
        exposure_time_seconds=_float(header.get("EXPTIME")),
        telescope=_text(header.get("TELESCOP")),
        instrument=_text(header.get("INSTRUME")),
        software_version=_text(header.get("VERSION")),
    )


# ---------------------------------------------------------------------- helpers
def pixel_scale_arcsec(wcs: WCS) -> float | None:
    try:
        scales = wcs.proj_plane_pixel_scales()
    except Exception:  # pragma: no cover - defensive
        return None
    values = [float(scale.to_value(u.arcsec)) for scale in scales]
    return sum(values) / len(values) if values else None


def _orientation_deg(wcs: WCS) -> float | None:
    try:
        first = wcs.pixel_to_world(0, 0)
        second = wcs.pixel_to_world(1, 0)
        return _round(float(first.position_angle(second).to_value(u.deg)), 4)
    except Exception:  # pragma: no cover - defensive
        return None


def _world_to_radec(wcs: WCS, x: float, y: float) -> tuple[float | None, float | None]:
    try:
        world = wcs.pixel_to_world(x, y)
        return _round(float(world.ra.to_value(u.deg)), 8), _round(float(world.dec.to_value(u.deg)), 8)
    except Exception:  # pragma: no cover - defensive
        return None, None


def _projection(ctype: list[str]) -> str | None:
    if not ctype:
        return None
    tail = ctype[0].split("---")[-1]
    return tail.split("-")[0] or None


def _text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _pair(first: object, second: object) -> list[float] | None:
    values = (_float(first), _float(second))
    return None if any(value is None for value in values) else [values[0], values[1]]  # type: ignore[list-item]


def _round(value: float | None, digits: int) -> float | None:
    return round(value, digits) if value is not None and math.isfinite(value) else None
