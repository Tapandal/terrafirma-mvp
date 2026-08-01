"""
geo_risk.py
===========

Earth Engine risk analytics for property underwriting and parametric insurance.

Two capabilities:

1. `stability_score()`   - Multi-year "pre-loss" volatility score (0-100) built on the
                            AlphaEarth Foundations Satellite Embedding dataset, with a
                            Sentinel-1/2 fallback for locations/years without embeddings.

2. `event_anomaly()`     - Change-detection: compares a pre-disaster baseline embedding
                            vector against a post-disaster vector and returns a distance
                            anomaly score normalised against that pixel's own historical
                            inter-annual variability.

3. `rapid_sar_anomaly()` / `rapid_optical_anomaly()`
                          - Near-real-time triggers. The embedding dataset is ANNUAL, so it
                            cannot fire a trigger days after an event. These functions cover
                            the operational window; the embedding-based score is the annual
                            confirmation / reserving signal.

Dataset notes
-------------
GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL: 64 bands (A00..A63), 10 m, 2017-2024 (2025 pending).
Each pixel vector is UNIT LENGTH, so:
    dot(a, b) == cosine similarity in [-1, 1]
    L2(a, b)  == sqrt(2 - 2*cos) in [0, 2]
This is why the distance maths below is cheap and bounded.

Requirements:  pip install earthengine-api
Auth:          earthengine authenticate  (or a service account, see initialize())
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import ee

__all__ = [
    "Config",
    "initialize",
    "stability_score",
    "event_anomaly",
    "rapid_sar_anomaly",
    "rapid_optical_anomaly",
    "underwrite_location",
    "assess_event",
    "calibrate_anchors",
]

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

EMBEDDING_ID = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
EMBEDDING_BANDS = [f"A{i:02d}" for i in range(64)]

S2_ID = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_SCORE_PLUS_ID = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
S1_ID = "COPERNICUS/S1_GRD"

_EPS = 1e-9


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """Tunable parameters.

    IMPORTANT — the *_lo / *_hi anchors are normalisation bounds, not physical constants.
    They map a raw distance onto 0-1 before weighting. The defaults below are sane
    starting points for mixed land cover, but you MUST recalibrate them against your own
    book of business before pricing anything. Use `calibrate_anchors()` to derive them
    empirically from a sample of your portfolio.
    """

    # --- geometry / sampling -----------------------------------------------------------
    radius_m: float = 500.0          # AOI = circular buffer around the point
    scale_m: float = 10.0            # native embedding resolution
    max_pixels: float = 1e9

    # --- embedding time range ----------------------------------------------------------
    first_year: int = 2017
    last_year: int = 2024

    # --- stability normalisation anchors -----------------------------------------------
    # Mean year-over-year L2 step distance. Stable forest/urban sits low; irrigated
    # cropland and floodplains sit high.
    step_lo: float = 0.10
    step_hi: float = 0.70
    # Dispersion of yearly vectors around their multi-year centroid.
    dispersion_lo: float = 0.08
    dispersion_hi: float = 0.55
    # Net drift from first to last year (structural conversion signal).
    drift_lo: float = 0.10
    drift_hi: float = 0.90

    # --- stability component weights (must sum to 1.0) ---------------------------------
    w_step: float = 0.30             # average churn
    w_tail: float = 0.20             # worst single-year jump (p90 across AOI)
    w_dispersion: float = 0.20       # spread around the norm
    w_regime: float = 0.30           # directional drift == land-use conversion

    # --- event anomaly -----------------------------------------------------------------
    min_baseline_years: int = 3      # need >=3 to estimate a per-pixel std dev
    sd_floor: float = 0.05           # guards against divide-by-tiny on very stable pixels
    z_critical: float = 3.0          # per-pixel z above which a pixel counts as "changed"
    # Areal damage index -> payout fraction. Parametric contracts want a step function
    # on an auditable index, not a continuous model output.
    payout_tiers: Tuple[Tuple[float, float], ...] = (
        (0.10, 0.25),
        (0.25, 0.50),
        (0.50, 1.00),
    )

    # --- rapid (near-real-time) triggers ------------------------------------------------
    pre_window_days: int = 90
    post_window_days: int = 20
    s1_db_threshold: float = 3.0     # |Δ dB| considered anomalous
    cloud_score_threshold: float = 0.60  # Cloud Score+ 'cs_cdf' keep-threshold

    def validate(self) -> None:
        total = self.w_step + self.w_tail + self.w_dispersion + self.w_regime
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Stability weights must sum to 1.0, got {total}")
        if self.min_baseline_years < 3:
            raise ValueError("Need at least 3 baseline years for a per-pixel std dev.")


DEFAULT_CONFIG = Config()


# --------------------------------------------------------------------------------------
# Initialisation & geometry
# --------------------------------------------------------------------------------------


def initialize(project: str, service_account_key: Optional[str] = None) -> None:
    """Initialise Earth Engine. `project` is your Cloud project with the EE API enabled."""
    if service_account_key:
        with open(service_account_key) as fh:
            email = json.load(fh)["client_email"]
        creds = ee.ServiceAccountCredentials(email, service_account_key)
        ee.Initialize(creds, project=project)
    else:
        ee.Initialize(project=project)


def _aoi(lat: float, lon: float, radius_m: float) -> ee.Geometry:
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError(f"Coordinates out of range: {lat}, {lon}")
    return ee.Geometry.Point([lon, lat]).buffer(radius_m)


# --------------------------------------------------------------------------------------
# Embedding primitives (all server-side)
# --------------------------------------------------------------------------------------


def _l2_normalize(img: ee.Image) -> ee.Image:
    """Re-project a vector image onto the unit hypersphere.

    Needed after any averaging: the mean of unit vectors is NOT unit length, and skipping
    this quietly biases every distance downstream.
    """
    norm = img.pow(2).reduce(ee.Reducer.sum()).sqrt().max(_EPS)
    return img.divide(norm)


def _embedding_for_year(year: int, aoi: ee.Geometry) -> ee.Image:
    """Mosaic of the annual embedding tiles intersecting the AOI."""
    col = (
        ee.ImageCollection(EMBEDDING_ID)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filterBounds(aoi)
    )
    return ee.Image(col.mosaic()).select(EMBEDDING_BANDS).set("year", year)


def _euclidean(a: ee.Image, b: ee.Image, name: str = "l2") -> ee.Image:
    """L2 distance between two 64-band vector images. Range [0, 2] for unit vectors."""
    return a.subtract(b).pow(2).reduce(ee.Reducer.sum()).sqrt().rename(name)


def _cosine_distance(a: ee.Image, b: ee.Image, name: str = "cosdist") -> ee.Image:
    """1 - cosine similarity. Range [0, 2]. Assumes both inputs are unit length."""
    dot = a.multiply(b).reduce(ee.Reducer.sum())
    return ee.Image.constant(1).subtract(dot).rename(name)


def available_embedding_years(aoi: ee.Geometry) -> List[int]:
    """Years with embedding coverage over this AOI. One round trip."""
    stamps = (
        ee.ImageCollection(EMBEDDING_ID)
        .filterBounds(aoi)
        .aggregate_array("system:time_start")
        .getInfo()
    ) or []
    return sorted({datetime.fromtimestamp(s / 1000, tz=timezone.utc).year for s in stamps})


def _reduce(img: ee.Image, aoi: ee.Geometry, reducer: ee.Reducer, cfg: Config) -> Dict:
    return img.reduceRegion(
        reducer=reducer,
        geometry=aoi,
        scale=cfg.scale_m,
        maxPixels=cfg.max_pixels,
        bestEffort=True,
    ).getInfo()


# --------------------------------------------------------------------------------------
# Scoring helpers
# --------------------------------------------------------------------------------------


def _rescale(value: Optional[float], lo: float, hi: float) -> float:
    """Clamp a raw metric to 0-1 against calibration anchors."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo + _EPS)))


def _band(stats: Dict, key: str) -> Optional[float]:
    v = stats.get(key)
    return float(v) if v is not None else None


# --------------------------------------------------------------------------------------
# 1. Multi-year stability score (pre-loss underwriting)
# --------------------------------------------------------------------------------------


def stability_score(
    lat: float,
    lon: float,
    cfg: Config = DEFAULT_CONFIG,
    years: Optional[Sequence[int]] = None,
) -> Dict:
    """Multi-year environmental volatility score for underwriting.

    Returns a 0-100 risk score where HIGHER = MORE VOLATILE = MORE RISK, plus the raw
    components so an underwriter can see *why*.

    The four components measure different things and should not be collapsed:

      mean_step     average year-over-year embedding distance. General churn.
      tail_step     90th percentile of the worst annual jump across the AOI. Captures a
                    localised disturbance that the mean would wash out.
      dispersion    spread of yearly vectors around their own centroid.
      regime_shift  net_drift * straightness. Straightness = |end - start| / path length.
                    This is the key discriminator: a rice paddy oscillates violently
                    year to year (high churn, straightness ~0) but always returns to the
                    same place. Deforestation or urban encroachment moves steadily in one
                    direction (straightness ~1) and never comes back. Only the second is
                    a durable change in the risk profile, so it is weighted separately.
    """
    cfg.validate()
    aoi = _aoi(lat, lon, cfg.radius_m)

    if years is None:
        available = available_embedding_years(aoi)
        years = [y for y in available if cfg.first_year <= y <= cfg.last_year]

    years = sorted(years)
    if len(years) < 3:
        return {
            "status": "insufficient_embedding_coverage",
            "years_available": years,
            "message": "Fewer than 3 embedding years; use stability_score_sar_optical().",
        }

    imgs = [_l2_normalize(_embedding_for_year(y, aoi)) for y in years]

    # Year-over-year steps.
    steps = [_euclidean(imgs[i], imgs[i + 1]) for i in range(len(imgs) - 1)]
    step_stack = ee.ImageCollection(steps)
    mean_step = step_stack.mean().rename("mean_step")
    max_step = step_stack.max().rename("max_step")
    path_length = step_stack.sum().rename("path_length")

    # Dispersion around the multi-year centroid.
    centroid = _l2_normalize(ee.ImageCollection(imgs).mean())
    dispersion = (
        ee.ImageCollection([_euclidean(im, centroid) for im in imgs]).mean().rename("dispersion")
    )

    # Directional drift vs. oscillation.
    net_drift = _euclidean(imgs[0], imgs[-1], "net_drift")
    straightness = net_drift.divide(path_length.max(_EPS)).clamp(0, 1).rename("straightness")
    regime_shift = net_drift.multiply(straightness).rename("regime_shift")

    metrics = ee.Image.cat(
        [mean_step, max_step, dispersion, net_drift, straightness, regime_shift]
    )

    reducer = ee.Reducer.mean().combine(
        ee.Reducer.percentile([90]), sharedInputs=True
    )
    stats = _reduce(metrics, aoi, reducer, cfg)

    raw = {
        "mean_step": _band(stats, "mean_step_mean"),
        "tail_step": _band(stats, "max_step_p90"),
        "dispersion": _band(stats, "dispersion_mean"),
        "net_drift": _band(stats, "net_drift_mean"),
        "straightness": _band(stats, "straightness_mean"),
        "regime_shift": _band(stats, "regime_shift_mean"),
    }

    components = {
        "churn": _rescale(raw["mean_step"], cfg.step_lo, cfg.step_hi),
        "tail": _rescale(raw["tail_step"], cfg.step_lo, cfg.step_hi),
        "dispersion": _rescale(raw["dispersion"], cfg.dispersion_lo, cfg.dispersion_hi),
        "regime": _rescale(raw["regime_shift"], cfg.drift_lo, cfg.drift_hi),
    }

    score = 100.0 * (
        cfg.w_step * components["churn"]
        + cfg.w_tail * components["tail"]
        + cfg.w_dispersion * components["dispersion"]
        + cfg.w_regime * components["regime"]
    )

    straight = raw["straightness"] or 0.0
    if straight > 0.6 and components["regime"] > 0.5:
        pattern = "directional_conversion"
    elif components["churn"] > 0.5 and straight < 0.35:
        pattern = "cyclical_volatility"
    elif components["churn"] < 0.25:
        pattern = "stable"
    else:
        pattern = "mixed"

    return {
        "status": "ok",
        "method": "alphaearth_embedding",
        "lat": lat,
        "lon": lon,
        "years": years,
        "stability_risk_score": round(score, 1),
        "risk_band": _risk_band(score),
        "pattern": pattern,
        "components_0_1": {k: round(v, 4) for k, v in components.items()},
        "raw_metrics": {k: (round(v, 5) if v is not None else None) for k, v in raw.items()},
    }


def _risk_band(score: float) -> str:
    if score < 20:
        return "very_low"
    if score < 40:
        return "low"
    if score < 60:
        return "moderate"
    if score < 80:
        return "elevated"
    return "high"


def stability_score_sar_optical(
    lat: float,
    lon: float,
    cfg: Config = DEFAULT_CONFIG,
    start_year: int = 2019,
    end_year: int = 2024,
) -> Dict:
    """Fallback stability score from Sentinel-1/2 directly.

    Use where embeddings are unavailable (pre-2017, polar gaps, or a year not yet
    published). Coarser and less semantically rich than the embedding route, but built
    from the same underlying sensors, so the two correlate reasonably.

    Components: interannual NDVI variability, NDVI trend magnitude, and S1 VH backscatter
    variability (structural/moisture change, cloud-independent).
    """
    aoi = _aoi(lat, lon, cfg.radius_m)
    start, end = f"{start_year}-01-01", f"{end_year + 1}-01-01"

    s2 = (
        ee.ImageCollection(S2_ID)
        .filterBounds(aoi)
        .filterDate(start, end)
        .linkCollection(ee.ImageCollection(CLOUD_SCORE_PLUS_ID), ["cs_cdf"])
        .map(lambda im: _mask_s2(im, cfg))
    )
    ndvi = s2.map(
        lambda im: im.normalizedDifference(["B8", "B4"]).rename("ndvi").copyProperties(im, ["system:time_start"])
    )

    ndvi_std = ndvi.reduce(ee.Reducer.stdDev()).rename("ndvi_std")
    ndvi_range = (
        ndvi.reduce(ee.Reducer.percentile([90]))
        .subtract(ndvi.reduce(ee.Reducer.percentile([10])))
        .rename("ndvi_range")
    )

    # Linear trend in NDVI (slope per year) - a proxy for directional conversion.
    def _with_t(im):
        t = ee.Number(im.date().difference(ee.Date(start), "year"))
        return im.addBands(ee.Image.constant(t).float().rename("t"))

    trend = (
        ndvi.map(_with_t)
        .select(["t", "ndvi"])
        .reduce(ee.Reducer.linearFit())
        .select("scale")
        .abs()
        .rename("ndvi_trend")
    )

    s1 = (
        ee.ImageCollection(S1_ID)
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select("VH")
    )
    vh_std = s1.reduce(ee.Reducer.stdDev()).rename("vh_std")

    metrics = ee.Image.cat([ndvi_std, ndvi_range, trend, vh_std])
    stats = _reduce(metrics, aoi, ee.Reducer.mean(), cfg)

    comp = {
        "ndvi_variability": _rescale(_band(stats, "ndvi_std"), 0.02, 0.25),
        "ndvi_range": _rescale(_band(stats, "ndvi_range"), 0.05, 0.60),
        "ndvi_trend": _rescale(_band(stats, "ndvi_trend"), 0.005, 0.08),
        "sar_variability": _rescale(_band(stats, "vh_std"), 0.5, 3.5),
    }
    score = 100.0 * (
        0.30 * comp["ndvi_variability"]
        + 0.20 * comp["ndvi_range"]
        + 0.30 * comp["ndvi_trend"]
        + 0.20 * comp["sar_variability"]
    )

    return {
        "status": "ok",
        "method": "sentinel1_sentinel2_fallback",
        "lat": lat,
        "lon": lon,
        "period": [start_year, end_year],
        "stability_risk_score": round(score, 1),
        "risk_band": _risk_band(score),
        "components_0_1": {k: round(v, 4) for k, v in comp.items()},
        "raw_metrics": stats,
    }


# --------------------------------------------------------------------------------------
# 2. Event change detection (parametric trigger)
# --------------------------------------------------------------------------------------


def event_anomaly(
    lat: float,
    lon: float,
    event_year: int,
    cfg: Config = DEFAULT_CONFIG,
    baseline_years: Optional[Sequence[int]] = None,
) -> Dict:
    """Compare a post-event embedding vector against a pre-event baseline.

    The core idea: a raw distance is meaningless on its own. Moving 0.4 in embedding space
    is catastrophic for a stable forest pixel and completely routine for a floodplain. So
    the distance is standardised against *that pixel's own* historical inter-annual
    variability:

        z = (d_post - mu_baseline) / max(sd_baseline, sd_floor)

    where mu/sd come from the distances of each baseline year to the baseline centroid.
    The result is comparable across land cover types and across a portfolio, which is what
    a parametric contract needs.

    The reported index is the AREAL FRACTION of the AOI exceeding z_critical, not the mean
    z. Mean z is easily dominated by a few extreme pixels; areal extent is what actually
    correlates with loss, and it is far easier to defend in a claims dispute.
    """
    cfg.validate()
    aoi = _aoi(lat, lon, cfg.radius_m)
    available = set(available_embedding_years(aoi))

    if baseline_years is None:
        baseline_years = [y for y in range(cfg.first_year, event_year) if y in available]
    baseline_years = sorted(baseline_years)

    if len(baseline_years) < cfg.min_baseline_years:
        raise ValueError(
            f"Need >= {cfg.min_baseline_years} baseline years, got {baseline_years}. "
            "Widen the baseline or use rapid_sar_anomaly() instead."
        )
    if event_year not in available:
        raise ValueError(
            f"No embedding for {event_year} (available: {sorted(available)}). "
            "Annual embeddings publish well after year-end - use the rapid_* functions "
            "for the operational trigger window."
        )

    base_imgs = [_l2_normalize(_embedding_for_year(y, aoi)) for y in baseline_years]
    baseline = _l2_normalize(ee.ImageCollection(base_imgs).mean())
    post = _l2_normalize(_embedding_for_year(event_year, aoi))

    # Null distribution: how far each baseline year sits from the baseline centroid.
    null_stack = ee.ImageCollection([_euclidean(im, baseline) for im in base_imgs])
    mu = null_stack.mean()
    sd = null_stack.reduce(ee.Reducer.stdDev()).max(cfg.sd_floor)

    d_post = _euclidean(post, baseline, "distance")
    cos_post = _cosine_distance(post, baseline, "cosine_distance")
    z = d_post.subtract(mu).divide(sd).rename("z")
    changed = z.gt(cfg.z_critical).rename("changed")

    metrics = ee.Image.cat([d_post, cos_post, z, changed, mu.rename("mu"), sd.rename("sd")])
    reducer = ee.Reducer.mean().combine(ee.Reducer.percentile([90, 95]), sharedInputs=True)
    stats = _reduce(metrics, aoi, reducer, cfg)

    damage_fraction = _band(stats, "changed_mean") or 0.0
    payout = _payout_fraction(damage_fraction, cfg.payout_tiers)

    return {
        "status": "ok",
        "method": "alphaearth_embedding_zscore",
        "lat": lat,
        "lon": lon,
        "event_year": event_year,
        "baseline_years": baseline_years,
        # --- the trigger index ---
        "damage_area_fraction": round(damage_fraction, 4),
        "trigger_fired": payout > 0.0,
        "payout_fraction": payout,
        # --- diagnostics ---
        "mean_l2_distance": _round(_band(stats, "distance_mean")),
        "p95_l2_distance": _round(_band(stats, "distance_p95")),
        "mean_cosine_distance": _round(_band(stats, "cosine_distance_mean")),
        "mean_z": _round(_band(stats, "z_mean")),
        "p90_z": _round(_band(stats, "z_p90")),
        "baseline_mu": _round(_band(stats, "mu_mean")),
        "baseline_sd": _round(_band(stats, "sd_mean")),
        "z_critical": cfg.z_critical,
    }


def _payout_fraction(index: float, tiers: Sequence[Tuple[float, float]]) -> float:
    payout = 0.0
    for threshold, fraction in sorted(tiers):
        if index >= threshold:
            payout = fraction
    return payout


def _round(v: Optional[float], nd: int = 5) -> Optional[float]:
    return round(v, nd) if v is not None else None


# --------------------------------------------------------------------------------------
# 3. Near-real-time triggers (operational window)
# --------------------------------------------------------------------------------------


def rapid_sar_anomaly(
    lat: float,
    lon: float,
    event_date: str,
    cfg: Config = DEFAULT_CONFIG,
) -> Dict:
    """Sentinel-1 log-ratio change detection. Cloud-independent, ~6-12 day revisit.

    Critically, this restricts the pre/post windows to the SAME relative orbit and pass
    direction. SAR backscatter is strongly geometry-dependent; comparing an ascending
    scene to a descending one produces large differences that have nothing to do with the
    event. Skipping this filter is the single most common source of false triggers.
    """
    aoi = _aoi(lat, lon, cfg.radius_m)
    event = ee.Date(event_date)
    pre_start = event.advance(-cfg.pre_window_days, "day")
    post_end = event.advance(cfg.post_window_days, "day")

    base = (
        ee.ImageCollection(S1_ID)
        .filterBounds(aoi)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH"])
    )

    pre_all = base.filterDate(pre_start, event)
    if pre_all.size().getInfo() == 0:
        return {"status": "no_pre_event_sar", "event_date": event_date}

    # Lock onto the most recent pre-event acquisition geometry.
    ref = ee.Image(pre_all.sort("system:time_start", False).first())
    orbit = ref.get("relativeOrbitNumber_start")
    direction = ref.get("orbitProperties_pass")
    geom_filter = ee.Filter.And(
        ee.Filter.eq("relativeOrbitNumber_start", orbit),
        ee.Filter.eq("orbitProperties_pass", direction),
    )

    pre = pre_all.filter(geom_filter)
    post = base.filterDate(event, post_end).filter(geom_filter)

    counts = ee.Dictionary({"pre": pre.size(), "post": post.size()}).getInfo()
    if counts["post"] == 0:
        return {
            "status": "no_post_event_sar",
            "event_date": event_date,
            "hint": "Widen post_window_days or relax the orbit filter.",
        }

    pre_img = pre.median()
    post_img = post.median()
    delta = post_img.subtract(pre_img).rename(["dVV", "dVH"])  # already dB -> log ratio

    magnitude = delta.pow(2).reduce(ee.Reducer.sum()).sqrt().rename("magnitude")
    changed = magnitude.gt(cfg.s1_db_threshold).rename("changed")

    stats = _reduce(
        ee.Image.cat([delta, magnitude, changed]),
        aoi,
        ee.Reducer.mean().combine(ee.Reducer.percentile([90]), sharedInputs=True),
        cfg,
    )

    fraction = _band(stats, "changed_mean") or 0.0
    return {
        "status": "ok",
        "method": "sentinel1_log_ratio",
        "event_date": event_date,
        "scene_counts": counts,
        "relative_orbit": orbit.getInfo() if hasattr(orbit, "getInfo") else orbit,
        "changed_area_fraction": round(fraction, 4),
        "payout_fraction": _payout_fraction(fraction, cfg.payout_tiers),
        "mean_dVV_db": _round(_band(stats, "dVV_mean"), 3),
        "mean_dVH_db": _round(_band(stats, "dVH_mean"), 3),
        "p90_magnitude_db": _round(_band(stats, "magnitude_p90"), 3),
    }


def rapid_optical_anomaly(
    lat: float,
    lon: float,
    event_date: str,
    cfg: Config = DEFAULT_CONFIG,
) -> Dict:
    """Sentinel-2 feature-vector change detection.

    Mirrors the embedding approach at lower dimension: build a per-pixel feature vector
    from reflectance plus indices, standardise each feature by its own pre-event mean and
    std, then take the Euclidean norm of the standardised difference. That is a
    diagonal-covariance Mahalanobis distance, which keeps noisy bands from dominating.
    """
    aoi = _aoi(lat, lon, cfg.radius_m)
    event = ee.Date(event_date)
    pre_start = event.advance(-cfg.pre_window_days, "day")
    post_end = event.advance(cfg.post_window_days, "day")

    col = (
        ee.ImageCollection(S2_ID)
        .filterBounds(aoi)
        .linkCollection(ee.ImageCollection(CLOUD_SCORE_PLUS_ID), ["cs_cdf"])
        .map(lambda im: _add_indices(_mask_s2(im, cfg)))
    )
    features = ["B4", "B8", "B11", "B12", "ndvi", "nbr", "ndwi"]

    pre = col.filterDate(pre_start, event).select(features)
    post = col.filterDate(event, post_end).select(features)

    counts = ee.Dictionary({"pre": pre.size(), "post": post.size()}).getInfo()
    if counts["pre"] < 2 or counts["post"] < 1:
        return {
            "status": "insufficient_clear_optical",
            "scene_counts": counts,
            "hint": "Cloud cover likely. Fall back to rapid_sar_anomaly().",
        }

    pre_mean = pre.mean()
    pre_sd = pre.reduce(ee.Reducer.stdDev()).rename(features).max(0.01)
    post_med = post.median()

    z = post_med.subtract(pre_mean).divide(pre_sd)
    magnitude = z.pow(2).reduce(ee.Reducer.sum()).sqrt().rename("magnitude")
    changed = magnitude.gt(cfg.z_critical).rename("changed")

    dnbr = pre_mean.select("nbr").subtract(post_med.select("nbr")).rename("dnbr")

    stats = _reduce(
        ee.Image.cat([magnitude, changed, dnbr]),
        aoi,
        ee.Reducer.mean().combine(ee.Reducer.percentile([90]), sharedInputs=True),
        cfg,
    )

    fraction = _band(stats, "changed_mean") or 0.0
    return {
        "status": "ok",
        "method": "sentinel2_standardised_vector_distance",
        "event_date": event_date,
        "scene_counts": counts,
        "changed_area_fraction": round(fraction, 4),
        "payout_fraction": _payout_fraction(fraction, cfg.payout_tiers),
        "mean_magnitude": _round(_band(stats, "magnitude_mean"), 3),
        "p90_magnitude": _round(_band(stats, "magnitude_p90"), 3),
        "mean_dnbr": _round(_band(stats, "dnbr_mean"), 4),
    }


def _mask_s2(img: ee.Image, cfg: Config) -> ee.Image:
    """Cloud Score+ masking and reflectance scaling.

    Cloud Score+ outperforms the older QA60/SCL approaches, especially on haze and
    cirrus. Note the harmonisation offset already applied by S2_SR_HARMONIZED for
    post-2022-01-25 scenes - do not add your own.
    """
    clear = img.select("cs_cdf").gte(cfg.cloud_score_threshold)
    return img.updateMask(clear).divide(10000).copyProperties(img, ["system:time_start"])


def _add_indices(img: ee.Image) -> ee.Image:
    return img.addBands(
        [
            img.normalizedDifference(["B8", "B4"]).rename("ndvi"),
            img.normalizedDifference(["B8", "B12"]).rename("nbr"),
            img.normalizedDifference(["B3", "B8"]).rename("ndwi"),
        ]
    )


# --------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------


def calibrate_anchors(
    points: Sequence[Tuple[float, float]],
    cfg: Config = DEFAULT_CONFIG,
    low_pct: float = 5.0,
    high_pct: float = 95.0,
) -> Dict:
    """Derive normalisation anchors from a reference sample of locations.

    Run this over a few hundred points representative of your book, then feed the results
    back into Config. Absolute embedding distances have no external units, so anchors set
    by intuition will misprice systematically. The defaults in Config are placeholders.
    """
    collected: Dict[str, List[float]] = {"mean_step": [], "dispersion": [], "regime_shift": []}
    for lat, lon in points:
        try:
            res = stability_score(lat, lon, cfg)
        except Exception:  # noqa: BLE001 - a bad point should not abort the sweep
            continue
        if res.get("status") != "ok":
            continue
        for key in collected:
            val = res["raw_metrics"].get(key)
            if val is not None:
                collected[key].append(val)

    def _pct(values: List[float], pct: float) -> Optional[float]:
        if not values:
            return None
        vals = sorted(values)
        idx = min(len(vals) - 1, max(0, int(round(pct / 100.0 * (len(vals) - 1)))))
        return vals[idx]

    return {
        "n_points_used": len(collected["mean_step"]),
        "suggested": {
            "step_lo": _pct(collected["mean_step"], low_pct),
            "step_hi": _pct(collected["mean_step"], high_pct),
            "dispersion_lo": _pct(collected["dispersion"], low_pct),
            "dispersion_hi": _pct(collected["dispersion"], high_pct),
            "drift_lo": _pct(collected["regime_shift"], low_pct),
            "drift_hi": _pct(collected["regime_shift"], high_pct),
        },
    }


# --------------------------------------------------------------------------------------
# Orchestrators
# --------------------------------------------------------------------------------------


def underwrite_location(lat: float, lon: float, cfg: Config = DEFAULT_CONFIG) -> Dict:
    """Full pre-bind assessment, with automatic fallback if embeddings are unavailable."""
    result = stability_score(lat, lon, cfg)
    if result.get("status") != "ok":
        result = stability_score_sar_optical(lat, lon, cfg)
    return result


def assess_event(
    lat: float,
    lon: float,
    event_date: str,
    cfg: Config = DEFAULT_CONFIG,
) -> Dict:
    """Post-event assessment across all three time horizons.

    Returns rapid triggers (days), the annual embedding confirmation (if published), and
    the pre-loss stability context needed to interpret the anomaly.
    """
    event_year = int(event_date[:4])
    out: Dict = {
        "lat": lat,
        "lon": lon,
        "event_date": event_date,
        "rapid_sar": rapid_sar_anomaly(lat, lon, event_date, cfg),
        "rapid_optical": rapid_optical_anomaly(lat, lon, event_date, cfg),
    }
    try:
        out["annual_embedding"] = event_anomaly(lat, lon, event_year, cfg)
    except ValueError as exc:
        out["annual_embedding"] = {"status": "unavailable", "reason": str(exc)}

    try:
        out["pre_loss_context"] = stability_score(lat, lon, cfg)
    except Exception as exc:  # noqa: BLE001
        out["pre_loss_context"] = {"status": "error", "reason": str(exc)}

    # Rapid SAR is the primary operational trigger: it is cloud-independent and available
    # within days. Optical corroborates when clear.
    sar = out["rapid_sar"]
    out["recommended_payout_fraction"] = (
        sar.get("payout_fraction", 0.0) if sar.get("status") == "ok" else None
    )
    return out


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(description="Earth Engine insurance risk analytics")
    parser.add_argument("--project", required=True, help="Google Cloud project ID")
    parser.add_argument("--key", default=None, help="Service account JSON key path")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--radius", type=float, default=500.0)
    parser.add_argument(
        "--mode", choices=["underwrite", "event"], default="underwrite"
    )
    parser.add_argument("--event-date", default=None, help="YYYY-MM-DD, required for --mode event")
    args = parser.parse_args()

    initialize(args.project, args.key)
    cfg = Config(radius_m=args.radius)

    if args.mode == "event":
        if not args.event_date:
            parser.error("--event-date is required with --mode event")
        result = assess_event(args.lat, args.lon, args.event_date, cfg)
    else:
        result = underwrite_location(args.lat, args.lon, cfg)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _main()
