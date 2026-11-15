"""Shared observation-artifact resolution for Phase 2 and Phase 3 services.

Given an observation id, either build a cached IRSA cutout around a target or
download the full product. This is the same policy the Phase 2 comparison uses,
so Phase 3 does not duplicate retrieval, trust checks or cache behaviour.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from app.core.exceptions import MissingAccessUrlError
from app.core.validation import validate_coordinates, validate_cutout_size
from app.services.fits_service import FitsProduct, download_fits, get_fits_product


@dataclass(frozen=True)
class ResolvedArtifact:
    detail: Any
    path: Path
    source: Literal["cutout", "product"]
    product: FitsProduct


def resolve_observation_artifact(
    observations: Any,
    observation_id: str,
    ra: float | None = None,
    dec: float | None = None,
    size: float | None = None,
    *,
    default_size: float,
    session: Any | None = None,
) -> ResolvedArtifact:
    """Resolve one observation to a FITS path without modifying the original.

    ``ra``/``dec`` request a cutout of ``size`` degrees; omitting them returns
    the full archive product. Trusted-URL validation, retries, size caps and the
    hash-addressed disk cache all come from the existing Phase 1/2 services.
    """
    detail = observations.get_observation(observation_id).observation
    if not detail.fits_access_url:
        raise MissingAccessUrlError(
            "This observation has no FITS access_url", details={"observation_id": observation_id}
        )
    product = get_fits_product(
        {
            "access_url": detail.fits_access_url,
            "access_format": detail.access_format,
            "obs_publisher_did": detail.obs_publisher_did,
            "observation_id": observation_id,
        },
        observations.settings,
    )
    http_session = session or observations.client.session

    if ra is not None and dec is not None:
        ra, dec = validate_coordinates(ra, dec)
        size = validate_cutout_size(
            size if size is not None else default_size, observations.settings.max_cutout_size_deg
        )
        artifact = observations.cutouts.fetch(product.access_url, ra, dec, size, session=http_session)
        return ResolvedArtifact(detail=detail, path=artifact.path, source="cutout", product=product)

    path = observations.settings.fits_dir / f"{hashlib.sha256(product.access_url.encode('utf-8')).hexdigest()}.fits"
    if not path.is_file():
        download_fits(product.access_url, path, session=http_session, settings=observations.settings)
    return ResolvedArtifact(detail=detail, path=path, source="product", product=product)


__all__ = ["ResolvedArtifact", "resolve_observation_artifact"]
