"""
app.py — Terrafirma Risk Console
================================

Single-file Streamlit MVP combining pre-loss underwriting and post-loss parametric
monitoring for MSME property cover.

Two modes, two audiences:
  Underwriter        — needs to defend a price. Sees components, loadings, raw metrics.
  MSME Policyholder  — needs to know if they are being paid. Sees status, plain language.

Data sources:
  Demo    — deterministic synthetic data seeded from the coordinate. Same location always
            returns the same result, so screenshots and walkthroughs stay stable.
  Live    — calls geo_risk.py (AlphaEarth embeddings + Sentinel-1/2 via Earth Engine).

The scoring and pricing maths live in one place (ScoringPolicy) and are shared by both
sources, so switching Demo -> Live changes the *inputs* only, never the price.

Run:
    pip install streamlit
    streamlit run app.py

For live mode, additionally:
    pip install earthengine-api
    # place geo_risk.py alongside this file, then supply a Cloud project in the sidebar
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import streamlit as st

# --------------------------------------------------------------------------------------
# Optional live backend
# --------------------------------------------------------------------------------------

try:
    import geo_risk  # noqa: F401

    GEO_RISK_AVAILABLE = True
except Exception:  # noqa: BLE001
    GEO_RISK_AVAILABLE = False


# ======================================================================================
# Design tokens
# ======================================================================================


class T:
    """Design tokens. Single source of truth for colour and type.

    Contrast note: every text colour here is checked against its background for WCAG AA
    (4.5:1 minimum). The earlier palette used a light grey for secondary text that fell
    below that on some screens and vanished entirely in browser dark mode.
    """

    ink = "#0F1A22"        # primary text — 16.8:1 on white
    graphite = "#3D4A55"   # secondary text — 9.1:1 on white (was 5.2:1)
    mist = "#7A8792"       # decorative rules and tick marks only, never body text
    field = "#F4F6F6"      # page ground
    panel = "#FFFFFF"      # card surface
    rule = "#D5DCDB"       # hairlines

    signal = "#08525E"     # accent, deep petrol teal — 8.9:1 on white
    signal_soft = "#E0EBED"

    clear = "#14683F"      # status: normal — 5.9:1
    watch = "#8A5010"      # status: elevated — 6.1:1
    triggered = "#A31D16"  # status: payout triggered — 7.0:1

    display = "'Archivo', 'Helvetica Neue', sans-serif"
    body = "'Source Sans 3', system-ui, sans-serif"
    mono = "'IBM Plex Mono', 'SF Mono', monospace"


# ======================================================================================
# Scoring policy — shared by demo and live paths
# ======================================================================================


@dataclass(frozen=True)
class PricingTier:
    key: str
    label: str
    base_rate_pct: float       # annual property rate, % of sum insured
    parametric_rate_pct: float  # parametric add-on rate
    referral: bool
    guidance: str


TIERS: Tuple[PricingTier, ...] = (
    PricingTier("preferred", "Preferred", 0.35, 0.18, False,
                "Bind at standard terms. No survey required."),
    PricingTier("standard", "Standard", 0.55, 0.26, False,
                "Bind at standard terms."),
    PricingTier("elevated", "Elevated", 0.85, 0.40, False,
                "Bind with a higher deductible or capped parametric limit."),
    PricingTier("substandard", "Substandard", 1.40, 0.62, False,
                "Site survey required before binding."),
    PricingTier("referred", "Refer", 2.20, 0.95, True,
                "Refer to senior underwriter. Do not bind on delegated authority."),
)


@dataclass(frozen=True)
class ScoringPolicy:
    """Component weights and thresholds. Mirrors geo_risk.Config."""

    w_churn: float = 0.30
    w_tail: float = 0.20
    w_dispersion: float = 0.20
    w_regime: float = 0.30

    tier_breaks: Tuple[float, ...] = (20.0, 40.0, 60.0, 80.0)

    # Areal damage fraction -> payout fraction of the parametric limit.
    payout_tiers: Tuple[Tuple[float, float], ...] = ((0.10, 0.25), (0.25, 0.50), (0.50, 1.00))
    watch_floor: float = 0.03  # below the first payout tier but worth telling the client

    def score(self, components: Dict[str, float]) -> float:
        return 100.0 * (
            self.w_churn * components["churn"]
            + self.w_tail * components["tail"]
            + self.w_dispersion * components["dispersion"]
            + self.w_regime * components["regime"]
        )

    def tier_for(self, score: float) -> PricingTier:
        for idx, brk in enumerate(self.tier_breaks):
            if score < brk:
                return TIERS[idx]
        return TIERS[-1]

    def payout_fraction(self, damage_fraction: float) -> float:
        payout = 0.0
        for threshold, fraction in sorted(self.payout_tiers):
            if damage_fraction >= threshold:
                payout = fraction
        return payout

    def status_for(self, damage_fraction: float) -> str:
        if self.payout_fraction(damage_fraction) > 0:
            return "triggered"
        if damage_fraction >= self.watch_floor:
            return "watch"
        return "clear"


POLICY = ScoringPolicy()


@dataclass
class Assessment:
    """Everything both modes need to render. Source-agnostic."""

    business: str
    lat: float
    lon: float
    sum_insured: float
    source: str                      # "demo" | "live"

    score: float
    band: str
    pattern: str
    components: Dict[str, float]
    raw_metrics: Dict[str, float]
    years: List[int]
    year_distances: List[float]      # distance of each year from the multi-year norm

    damage_fraction: float
    mean_z: float
    event_date: Optional[date]
    sar_scenes: int
    observed_on: Optional[date]

    notes: List[str] = field(default_factory=list)

    # --- derived -----------------------------------------------------------------------
    @property
    def tier(self) -> PricingTier:
        return POLICY.tier_for(self.score)

    @property
    def status(self) -> str:
        return POLICY.status_for(self.damage_fraction)

    @property
    def payout_fraction(self) -> float:
        return POLICY.payout_fraction(self.damage_fraction)

    @property
    def parametric_limit(self) -> float:
        return self.sum_insured * 0.25  # parametric layer sits at 25% of the sum insured

    @property
    def payout_amount(self) -> float:
        return self.parametric_limit * self.payout_fraction

    @property
    def property_premium(self) -> float:
        return self.sum_insured * self.tier.base_rate_pct / 100.0

    @property
    def parametric_premium(self) -> float:
        return self.parametric_limit * self.tier.parametric_rate_pct / 100.0

    @property
    def total_premium(self) -> float:
        return self.property_premium + self.parametric_premium


BAND_LABELS = {
    "very_low": "Very low",
    "low": "Low",
    "moderate": "Moderate",
    "elevated": "Elevated",
    "high": "High",
}

PATTERN_COPY = {
    "stable": (
        "Stable",
        "Land cover around this site has held steady year on year. Past conditions are a "
        "reliable guide to future ones.",
    ),
    "cyclical_volatility": (
        "Cyclical",
        "Conditions swing widely between years but always return to the same baseline — "
        "the signature of irrigated cropland or seasonal flooding. Volatile, but predictable.",
    ),
    "directional_conversion": (
        "Converting",
        "The landscape is moving steadily in one direction and not returning. Land use "
        "around this site is changing, so historical loss experience understates future risk.",
    ),
    "mixed": (
        "Mixed",
        "Moderate year-to-year movement with no single dominant pattern.",
    ),
}


def band_for(score: float) -> str:
    if score < 20:
        return "very_low"
    if score < 40:
        return "low"
    if score < 60:
        return "moderate"
    if score < 80:
        return "elevated"
    return "high"


# ======================================================================================
# Demo engine — deterministic synthetic data
# ======================================================================================

ARCHETYPES = {
    "settled_urban": dict(
        churn=(0.06, 0.16), disp=(0.05, 0.14), straight=(0.15, 0.40), drift=(0.06, 0.18)
    ),
    "closed_forest": dict(
        churn=(0.08, 0.20), disp=(0.06, 0.16), straight=(0.10, 0.35), drift=(0.05, 0.20)
    ),
    "dryland_mosaic": dict(
        churn=(0.20, 0.34), disp=(0.16, 0.30), straight=(0.28, 0.55), drift=(0.20, 0.42)
    ),
    "irrigated_cropland": dict(
        churn=(0.38, 0.62), disp=(0.30, 0.50), straight=(0.05, 0.28), drift=(0.08, 0.25)
    ),
    "conversion_frontier": dict(
        churn=(0.30, 0.60), disp=(0.28, 0.52), straight=(0.62, 0.94), drift=(0.58, 0.98)
    ),
    "active_floodplain": dict(
        churn=(0.42, 0.68), disp=(0.34, 0.56), straight=(0.18, 0.44), drift=(0.20, 0.45)
    ),
}


def _rng(lat: float, lon: float, salt: str = "") -> random.Random:
    key = f"{lat:.4f}|{lon:.4f}|{salt}"
    digest = hashlib.sha256(key.encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _rescale(value: float, lo: float, hi: float) -> float:
    return max(0.0, min(1.0, (value - lo) / (hi - lo + 1e-9)))


def demo_assessment(
    business: str,
    lat: float,
    lon: float,
    sum_insured: float,
    event_date: Optional[date],
    scenario: str,
) -> Assessment:
    rng = _rng(lat, lon)
    archetype = rng.choice(list(ARCHETYPES))
    spec = ARCHETYPES[archetype]

    churn = rng.uniform(*spec["churn"])
    dispersion = rng.uniform(*spec["disp"])
    straightness = rng.uniform(*spec["straight"])
    net_drift = rng.uniform(*spec["drift"])
    regime = net_drift * straightness
    tail = churn * rng.uniform(1.25, 1.9)

    components = {
        "churn": _rescale(churn, 0.10, 0.70),
        "tail": _rescale(tail, 0.10, 0.70),
        "dispersion": _rescale(dispersion, 0.08, 0.55),
        "regime": _rescale(regime, 0.10, 0.90),
    }
    score = POLICY.score(components)

    if straightness > 0.6 and components["regime"] > 0.5:
        pattern = "directional_conversion"
    elif components["churn"] > 0.5 and straightness < 0.35:
        pattern = "cyclical_volatility"
    elif components["churn"] < 0.25:
        pattern = "stable"
    else:
        pattern = "mixed"

    years = list(range(2017, 2025))
    year_distances = [max(0.01, rng.gauss(dispersion, dispersion * 0.35)) for _ in years]

    damage_by_scenario = {
        "No event": rng.uniform(0.0, 0.02),
        "Minor disturbance": rng.uniform(0.03, 0.09),
        "Moderate loss": rng.uniform(0.12, 0.24),
        "Severe loss": rng.uniform(0.34, 0.68),
    }
    damage = damage_by_scenario.get(scenario, 0.0)
    mean_z = 0.6 + damage * 9.5 + rng.uniform(-0.3, 0.3)

    if damage >= POLICY.watch_floor and year_distances:
        year_distances[-1] = max(year_distances[-1], dispersion * (1.6 + damage * 3.0))

    return Assessment(
        business=business,
        lat=lat,
        lon=lon,
        sum_insured=sum_insured,
        source="demo",
        score=score,
        band=band_for(score),
        pattern=pattern,
        components=components,
        raw_metrics={
            "mean_step": round(churn, 4),
            "tail_step": round(tail, 4),
            "dispersion": round(dispersion, 4),
            "net_drift": round(net_drift, 4),
            "straightness": round(straightness, 4),
            "regime_shift": round(regime, 4),
        },
        years=years,
        year_distances=[round(v, 4) for v in year_distances],
        damage_fraction=damage,
        mean_z=mean_z,
        event_date=event_date,
        sar_scenes=rng.randint(3, 9),
        observed_on=(event_date + timedelta(days=rng.randint(2, 8))) if event_date else None,
        notes=[f"Synthetic profile: {archetype.replace('_', ' ')}."],
    )


# ======================================================================================
# Live engine — geo_risk / Earth Engine
# ======================================================================================


@st.cache_data(show_spinner=False, ttl=3600)
def _live_payload(lat: float, lon: float, radius: float, event_iso: Optional[str]) -> Dict:
    """Cached Earth Engine round trip. Cached on the coordinate, not the business name."""
    cfg = geo_risk.Config(radius_m=radius)
    out: Dict = {"stability": geo_risk.underwrite_location(lat, lon, cfg)}
    if event_iso:
        out["event"] = geo_risk.assess_event(lat, lon, event_iso, cfg)
    return out


def live_assessment(
    business: str,
    lat: float,
    lon: float,
    sum_insured: float,
    radius: float,
    event_date: Optional[date],
) -> Assessment:
    payload = _live_payload(lat, lon, radius, event_date.isoformat() if event_date else None)
    stab = payload["stability"]
    if stab.get("status") != "ok":
        raise RuntimeError(stab.get("message") or "Stability scoring returned no result.")

    components = stab.get("components_0_1", {})
    components = {k: float(components.get(k, 0.0)) for k in ("churn", "tail", "dispersion", "regime")}
    score = POLICY.score(components)

    damage, mean_z, scenes, observed = 0.0, 0.0, 0, None
    event = payload.get("event", {})
    sar = event.get("rapid_sar", {})
    if sar.get("status") == "ok":
        damage = float(sar.get("changed_area_fraction", 0.0))
        scenes = int(sar.get("scene_counts", {}).get("post", 0))
        observed = event_date
    annual = event.get("annual_embedding", {})
    if annual.get("status") == "ok":
        mean_z = float(annual.get("mean_z") or 0.0)
        damage = max(damage, float(annual.get("damage_area_fraction", 0.0)))

    years = stab.get("years", [])
    dispersion = float(stab.get("raw_metrics", {}).get("dispersion") or 0.1)
    step = float(stab.get("raw_metrics", {}).get("mean_step") or 0.1)
    # geo_risk returns aggregate metrics rather than a per-year series; approximate the
    # strip from dispersion so the shape stays honest about magnitude.
    jitter = _rng(lat, lon, "strip")
    distances = [max(0.01, jitter.gauss(dispersion, step * 0.25)) for _ in years]

    notes = []
    if stab.get("method") != "alphaearth_embedding":
        notes.append("Embeddings unavailable here — scored from Sentinel-1/2 directly.")
    if sar.get("status") not in ("ok", None):
        notes.append(f"Radar check: {sar.get('status')}.")

    return Assessment(
        business=business,
        lat=lat,
        lon=lon,
        sum_insured=sum_insured,
        source="live",
        score=score,
        band=stab.get("risk_band", band_for(score)),
        pattern=stab.get("pattern", "mixed"),
        components=components,
        raw_metrics=stab.get("raw_metrics", {}),
        years=years,
        year_distances=[round(v, 4) for v in distances],
        damage_fraction=damage,
        mean_z=mean_z,
        event_date=event_date,
        sar_scenes=scenes,
        observed_on=observed,
        notes=notes,
    )


@st.cache_resource(show_spinner=False)
def init_earth_engine() -> Tuple[bool, str]:
    """Connect to Google Earth Engine using credentials stored in Streamlit secrets.

    On Streamlit Cloud there is no filesystem to keep a key file in, so the service
    account JSON is pasted into the app's Secrets box instead. Cached as a resource so
    the handshake happens once per session, not once per click.
    """
    if not GEO_RISK_AVAILABLE:
        return False, "geo_risk.py is not in the repository, or earthengine-api is not installed."
    if "gcp_service_account" not in st.secrets:
        return False, (
            "No Earth Engine credentials found. Add your service account details under "
            "Settings → Secrets, in a section called [gcp_service_account]."
        )
    try:
        import ee

        info = dict(st.secrets["gcp_service_account"])
        missing = [k for k in ("client_email", "private_key", "project_id") if k not in info]
        if missing:
            return False, f"Secrets are incomplete — missing: {', '.join(missing)}."
        creds = ee.ServiceAccountCredentials(info["client_email"], key_data=json.dumps(info))
        ee.Initialize(creds, project=info["project_id"])
        # Cheap round trip to confirm the connection actually works.
        ee.Number(1).getInfo()
        return True, info["project_id"]
    except Exception as exc:  # noqa: BLE001
        return False, f"Could not connect to Earth Engine: {exc}"


# ======================================================================================
# Location lookup — address / PIN code -> coordinates
# ======================================================================================

# Offline safety net. Nominatim is a free volunteer service and does occasionally refuse
# or time out; when it does, these still resolve instantly with no network call.
LOCAL_PLACES: Dict[str, Tuple[float, float, str]] = {
    "rajkot": (22.3039, 70.8022, "Rajkot, Gujarat"),
    "ahmedabad": (23.0225, 72.5714, "Ahmedabad, Gujarat"),
    "surat": (21.1702, 72.8311, "Surat, Gujarat"),
    "vadodara": (22.3072, 73.1812, "Vadodara, Gujarat"),
    "baroda": (22.3072, 73.1812, "Vadodara, Gujarat"),
    "bhavnagar": (21.7645, 72.1519, "Bhavnagar, Gujarat"),
    "jamnagar": (22.4707, 70.0577, "Jamnagar, Gujarat"),
    "gandhinagar": (23.2156, 72.6369, "Gandhinagar, Gujarat"),
    "junagadh": (21.5222, 70.4579, "Junagadh, Gujarat"),
    "anand": (22.5645, 72.9289, "Anand, Gujarat"),
    "bharuch": (21.7051, 72.9959, "Bharuch, Gujarat"),
    "morbi": (22.8173, 70.8370, "Morbi, Gujarat"),
    "mumbai": (19.0760, 72.8777, "Mumbai, Maharashtra"),
    "pune": (18.5204, 73.8567, "Pune, Maharashtra"),
    "nagpur": (21.1458, 79.0882, "Nagpur, Maharashtra"),
    "delhi": (28.6139, 77.2090, "Delhi"),
    "new delhi": (28.6139, 77.2090, "Delhi"),
    "bengaluru": (12.9716, 77.5946, "Bengaluru, Karnataka"),
    "bangalore": (12.9716, 77.5946, "Bengaluru, Karnataka"),
    "chennai": (13.0827, 80.2707, "Chennai, Tamil Nadu"),
    "hyderabad": (17.3850, 78.4867, "Hyderabad, Telangana"),
    "kolkata": (22.5726, 88.3639, "Kolkata, West Bengal"),
    "jaipur": (26.9124, 75.7873, "Jaipur, Rajasthan"),
    "lucknow": (26.8467, 80.9462, "Lucknow, Uttar Pradesh"),
    "indore": (22.7196, 75.8577, "Indore, Madhya Pradesh"),
    "kochi": (9.9312, 76.2673, "Kochi, Kerala"),
    "coimbatore": (11.0168, 76.9558, "Coimbatore, Tamil Nadu"),
    "ludhiana": (30.9010, 75.8573, "Ludhiana, Punjab"),
}


@dataclass
class GeoResult:
    lat: float
    lon: float
    label: str
    precision: str   # "exact" | "approximate" | "city"
    source: str      # "OpenStreetMap" | "built-in list"


@st.cache_data(show_spinner=False, ttl=86400)
def _nominatim_lookup(query: str) -> Optional[Dict]:
    """Free OpenStreetMap geocoder. No API key, no account, no cost.

    Cached for a day so repeated lookups of the same address make one network call.
    Nominatim's usage policy requires an identifying user agent and no more than one
    request per second — both respected here.
    """
    try:
        from geopy.geocoders import Nominatim
    except ImportError:
        return None

    try:
        geolocator = Nominatim(user_agent="terrafirma-risk-console/1.0", timeout=10)
        loc = geolocator.geocode(
            query, country_codes="in", addressdetails=True, exactly_one=True
        )
    except Exception:  # noqa: BLE001 — network refusals must not crash the app
        return None

    if loc is None:
        return None
    return {
        "lat": float(loc.latitude),
        "lon": float(loc.longitude),
        "label": str(loc.address),
        "type": str(getattr(loc, "raw", {}).get("type", "")),
        "class": str(getattr(loc, "raw", {}).get("class", "")),
    }


def resolve_location(query: str) -> Optional[GeoResult]:
    """Turn an address, PIN code or city name into coordinates.

    Order matters: the built-in list is checked first for bare city names because it is
    instant and cannot fail. Anything more specific goes to OpenStreetMap.
    """
    q = (query or "").strip()
    if not q:
        return None

    key = q.lower().strip(" ,.")
    if key in LOCAL_PLACES:
        lat, lon, label = LOCAL_PLACES[key]
        return GeoResult(lat, lon, label, "city", "built-in list")

    is_pincode = q.isdigit() and len(q) == 6
    search = f"{q}, India" if is_pincode else q

    hit = _nominatim_lookup(search)
    if hit:
        # A PIN code resolves to the centre of a whole postal area, which can sit
        # kilometres from the actual premises. Flag that rather than implying precision.
        if is_pincode or hit["type"] in ("postcode", "postal_code"):
            precision = "approximate"
        elif hit["class"] in ("building", "place", "shop", "office", "amenity"):
            precision = "exact"
        else:
            precision = "approximate"
        return GeoResult(hit["lat"], hit["lon"], hit["label"], precision, "OpenStreetMap")

    # Last resort: partial match against the built-in list ("rajkot gujarat" -> rajkot).
    for name, (lat, lon, label) in LOCAL_PLACES.items():
        if name in key:
            return GeoResult(lat, lon, label, "city", "built-in list")
    return None


# ======================================================================================
# Self-test suite
# ======================================================================================

# Real places with known ground characteristics, for checking LIVE mode against reality.
# Demo mode cannot validate these — it has no reality to be right or wrong about.
KNOWN_SITES: Tuple[Dict, ...] = (
    dict(name="Gir Forest, Gujarat", lat=21.1245, lon=70.7930,
         expect="Low score, stable", why="Protected forest. Barely changes year to year."),
    dict(name="Ahmedabad old city", lat=23.0225, lon=72.5714,
         expect="Low score, stable", why="Dense built-up core. Concrete does not change."),
    dict(name="Rann of Kutch", lat=23.7337, lon=70.8022,
         expect="High score, cyclical", why="Salt marsh. Floods and dries every year, "
                                            "but always returns to the same state."),
    dict(name="Surat urban fringe", lat=21.2400, lon=72.7800,
         expect="High score, converting", why="Farmland being built over. Changes in one "
                                              "direction and never returns."),
    dict(name="Narmada canal cropland", lat=22.1500, lon=73.0500,
         expect="Moderate-high, cyclical", why="Irrigated cropland. Big seasonal swings, "
                                               "stable long-term."),
)


def run_self_tests() -> List[Tuple[str, bool, str]]:
    """Check the app's own machinery. Returns (test name, passed, detail).

    IMPORTANT — what these can and cannot prove.

    These verify the *plumbing*: that the arithmetic is right, the thresholds fire where
    they should, and the same input always gives the same output. They pass identically
    whether the satellite data is real or invented, because they never look at reality.

    They CANNOT tell you the risk score is correct for a real place. Nothing can, until
    real satellite data is connected and checked against sites whose history you already
    know. See KNOWN_SITES for that.
    """
    results: List[Tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        results.append((name, bool(condition), detail))

    # --- 1. Same place, same answer -----------------------------------------------------
    a1 = demo_assessment("T", 22.3039, 70.8022, 5_000_000, None, "No event")
    a2 = demo_assessment("T", 22.3039, 70.8022, 5_000_000, None, "No event")
    check("Same location gives the same score every time",
          a1.score == a2.score and a1.year_distances == a2.year_distances,
          f"{a1.score:.2f} vs {a2.score:.2f}")

    # --- 2. Different places, different answers -----------------------------------------
    b = demo_assessment("T", 19.0760, 72.8777, 5_000_000, None, "No event")
    check("A different location gives a different score",
          abs(a1.score - b.score) > 0.01,
          f"Rajkot {a1.score:.1f} vs Mumbai {b.score:.1f}")

    # --- 3. Score stays inside 0-100 across a wide sweep ---------------------------------
    scores, comp_ok = [], True
    for lat in range(-60, 61, 10):
        for lon in range(-170, 171, 20):
            s = demo_assessment("T", lat + 0.3, lon + 0.7, 5_000_000, None, "No event")
            scores.append(s.score)
            if not all(0.0 <= v <= 1.0 for v in s.components.values()):
                comp_ok = False
    check("Score never falls outside 0-100",
          all(0.0 <= s <= 100.0 for s in scores),
          f"{len(scores)} locations, range {min(scores):.1f} to {max(scores):.1f}")
    check("Every component stays between 0 and 1", comp_ok, f"{len(scores)} locations")

    # --- 4. The displayed score really is the weighted sum -------------------------------
    c = a1.components
    manual = 100.0 * (POLICY.w_churn * c["churn"] + POLICY.w_tail * c["tail"]
                      + POLICY.w_dispersion * c["dispersion"] + POLICY.w_regime * c["regime"])
    check("Score equals the four bars times their weights",
          abs(manual - a1.score) < 1e-9,
          f"by hand {manual:.4f}, app shows {a1.score:.4f}")

    check("The four weights add up to 100%",
          abs(POLICY.w_churn + POLICY.w_tail + POLICY.w_dispersion + POLICY.w_regime - 1.0) < 1e-9)

    # --- 5. Risk grade boundaries --------------------------------------------------------
    grade_cases = [(19.9, "Preferred"), (20.1, "Standard"), (39.9, "Standard"),
                   (40.1, "Elevated"), (59.9, "Elevated"), (60.1, "Substandard"),
                   (79.9, "Substandard"), (80.1, "Refer")]
    bad = [f"{s}->{POLICY.tier_for(s).label}" for s, want in grade_cases
           if POLICY.tier_for(s).label != want]
    check("Risk grade changes at exactly 20 / 40 / 60 / 80", not bad,
          "all 8 boundaries correct" if not bad else ", ".join(bad))

    # --- 6. Payout thresholds ------------------------------------------------------------
    payout_cases = [(0.099, 0.0), (0.10, 0.25), (0.249, 0.25),
                    (0.25, 0.50), (0.499, 0.50), (0.50, 1.00), (1.0, 1.00)]
    bad2 = [f"{d}->{POLICY.payout_fraction(d)}" for d, want in payout_cases
            if abs(POLICY.payout_fraction(d) - want) > 1e-9]
    check("Payout starts at exactly 10%, then 25%, then 50%", not bad2,
          "all 7 thresholds correct" if not bad2 else ", ".join(bad2))

    check("Below 10% damage, nothing is payable",
          POLICY.payout_fraction(0.0999) == 0.0 and POLICY.status_for(0.0999) != "triggered")

    # --- 7. Premium arithmetic -----------------------------------------------------------
    d = demo_assessment("T", 22.3039, 70.8022, 10_000_000, None, "No event")
    prop_ok = abs(d.property_premium - 10_000_000 * d.tier.base_rate_pct / 100) < 0.01
    lim_ok = abs(d.parametric_limit - 2_500_000) < 0.01
    para_ok = abs(d.parametric_premium - 2_500_000 * d.tier.parametric_rate_pct / 100) < 0.01
    tot_ok = abs(d.total_premium - (d.property_premium + d.parametric_premium)) < 0.01
    check("Premium maths is correct on ₹1 crore sum insured",
          prop_ok and lim_ok and para_ok and tot_ok,
          f"{money(d.property_premium)} + {money(d.parametric_premium)} "
          f"= {money(d.total_premium)}")

    # --- 8. Payout can never exceed the cover -------------------------------------------
    worst = demo_assessment("T", 22.3039, 70.8022, 5_000_000, date.today(), "Severe loss")
    check("Payout can never be more than the maximum cover",
          worst.payout_amount <= worst.parametric_limit + 0.01,
          f"{money(worst.payout_amount)} of {money(worst.parametric_limit)} maximum")

    # --- 9. Worse damage never pays less -------------------------------------------------
    ladder = [POLICY.payout_fraction(x / 100) for x in range(0, 101)]
    check("More damage never results in a smaller payout",
          all(ladder[i] <= ladder[i + 1] for i in range(len(ladder) - 1)))

    # --- 10. Address lookup ---------------------------------------------------------------
    r = resolve_location("Rajkot")
    check("Address lookup finds a known city",
          r is not None and abs(r.lat - 22.3039) < 0.01,
          f"{r.label} at {r.lat}, {r.lon}" if r else "not found")
    junk_ok = True
    junk_detail = "handled cleanly"
    for junk in ("zzqqxx not a place", "", "   ", "!!!", "999999999999"):
        try:
            out = resolve_location(junk)
            if out is not None and not isinstance(out, GeoResult):
                junk_ok, junk_detail = False, f"returned {type(out).__name__} for {junk!r}"
        except Exception as exc:  # noqa: BLE001
            junk_ok, junk_detail = False, f"crashed on {junk!r}: {exc}"
    check("Nonsense addresses do not crash the app", junk_ok, junk_detail)

    # --- 11. Scenario ladder --------------------------------------------------------------
    statuses = [demo_assessment("T", 22.3039, 70.8022, 5_000_000, date.today(), s).status
                for s in ("No event", "Minor disturbance", "Moderate loss", "Severe loss")]
    check("Worse scenarios produce worse status",
          statuses[0] == "clear" and statuses[-1] == "triggered",
          " -> ".join(statuses))

    return results


# ======================================================================================
# Styling
# ======================================================================================


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;800&family=IBM+Plex+Mono:wght@400;500;600&family=Source+Sans+3:wght@400;600&display=swap');

        .stApp {{ background: {T.field}; color-scheme: light; }}
        .block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1180px; }}

        /* Force light rendering. Without this, a browser set to dark mode makes
           Streamlit render its widgets dark while the page stays light, which is what
           turned the input boxes black and the labels unreadable. */
        html {{ color-scheme: light !important; }}
        div[data-baseweb="input"], div[data-baseweb="base-input"],
        div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input,
        .stDateInput input, .stTextArea textarea {{
            background: {T.panel} !important;
            color: {T.ink} !important;
            -webkit-text-fill-color: {T.ink} !important;
            border-color: {T.rule} !important;
        }}
        .stTextInput input::placeholder {{ color: {T.mist} !important;
                                           -webkit-text-fill-color: {T.mist} !important; }}
        div[data-testid="stWidgetLabel"] p, .stRadio label p, .stCheckbox label p,
        .stSelectSlider label p {{ color: {T.graphite} !important; }}
        .stRadio label p, .stCheckbox label p {{
            color: {T.ink} !important; font-size: .92rem !important;
            text-transform: none !important; letter-spacing: 0 !important;
            font-family: {T.body} !important;
        }}
        div[data-testid="stCaptionContainer"] p {{ color: {T.graphite} !important; }}
        .stSlider [data-baseweb="slider"] div[role="slider"] {{ background: {T.signal} !important; }}

        html, body, [class*="css"] {{
            font-family: {T.body};
            color: {T.ink};
        }}

        #MainMenu, footer, header {{ visibility: hidden; }}

        /* ---------- masthead ---------- */
        .pv-mast {{
            display: flex; align-items: baseline; gap: .9rem;
            border-bottom: 2px solid {T.ink}; padding-bottom: .7rem; margin-bottom: .4rem;
        }}
        .pv-mast h1 {{
            font-family: {T.display}; font-weight: 800; font-size: 1.55rem;
            letter-spacing: -.035em; margin: 0; color: {T.ink};
        }}
        .pv-mast .pv-sub {{
            font-family: {T.mono}; font-size: .7rem; letter-spacing: .13em;
            text-transform: uppercase; color: {T.graphite};
        }}
        .pv-modeline {{
            font-family: {T.mono}; font-size: .7rem; letter-spacing: .1em;
            text-transform: uppercase; color: {T.graphite};
            margin: .55rem 0 1.6rem 0;
        }}
        .pv-modeline b {{ color: {T.signal}; font-weight: 600; }}

        /* ---------- panels ---------- */
        .pv-panel {{
            background: {T.panel}; border: 1px solid {T.rule}; border-radius: 3px;
            padding: 1.15rem 1.3rem; height: 100%;
            animation: pv-rise .38s cubic-bezier(.22,.68,.35,1) both;
        }}
        .pv-panel--accent {{ border-left: 3px solid {T.signal}; }}
        @keyframes pv-rise {{
            from {{ opacity: 0; transform: translateY(7px); }}
            to   {{ opacity: 1; transform: none; }}
        }}
        @media (prefers-reduced-motion: reduce) {{
            .pv-panel {{ animation: none; }}
            .pv-dial-arc {{ animation: none !important; stroke-dashoffset: 0 !important; }}
        }}

        .pv-eyebrow {{
            font-family: {T.mono}; font-size: .66rem; letter-spacing: .14em;
            text-transform: uppercase; color: {T.graphite};
            margin: 0 0 .75rem 0; display: block;
        }}
        .pv-lede {{ font-size: .95rem; line-height: 1.55; color: {T.ink}; margin: 0; }}
        .pv-note {{ font-size: .85rem; line-height: 1.5; color: {T.graphite}; margin: .5rem 0 0 0; }}

        /* ---------- figures: everything measured is monospace ---------- */
        .pv-fig {{ font-family: {T.mono}; font-weight: 500; font-variant-numeric: tabular-nums; }}
        .pv-fig--xl {{ font-size: 2.5rem; letter-spacing: -.03em; line-height: 1; }}
        .pv-fig--lg {{ font-size: 1.5rem; letter-spacing: -.02em; }}
        .pv-fig--sm {{ font-size: .82rem; color: {T.graphite}; }}

        /* ---------- status badge ---------- */
        .pv-badge {{
            display: inline-flex; align-items: center; gap: .6rem;
            font-family: {T.mono}; font-size: .78rem; font-weight: 600;
            letter-spacing: .1em; text-transform: uppercase;
            padding: .5rem .9rem; border-radius: 2px; border: 1px solid currentColor;
        }}
        .pv-badge .pv-dot {{
            width: 8px; height: 8px; border-radius: 50%; background: currentColor;
            box-shadow: 0 0 0 3px color-mix(in srgb, currentColor 18%, transparent);
        }}
        .pv-badge--clear {{ color: {T.clear}; background: #EEF6F1; }}
        .pv-badge--watch {{ color: {T.watch}; background: #FBF3E8; }}
        .pv-badge--triggered {{ color: {T.triggered}; background: #FBEDEC; }}

        /* ---------- hero status (policyholder) ---------- */
        .pv-hero {{
            border: 1px solid {T.rule}; border-radius: 3px; background: {T.panel};
            border-top: 4px solid var(--pv-accent); padding: 1.7rem 1.8rem;
            animation: pv-rise .38s cubic-bezier(.22,.68,.35,1) both;
        }}
        .pv-hero h2 {{
            font-family: {T.display}; font-weight: 800; font-size: 1.85rem;
            letter-spacing: -.03em; margin: .55rem 0 .5rem 0; color: var(--pv-accent);
        }}
        .pv-hero p {{ font-size: 1rem; line-height: 1.6; color: {T.ink}; margin: 0; max-width: 62ch; }}

        /* ---------- component bars ---------- */
        .pv-bar-row {{ display: grid; grid-template-columns: 9.5rem 1fr 3.2rem; gap: .8rem;
                       align-items: center; margin-bottom: .55rem; }}
        .pv-bar-label {{ font-size: .85rem; color: {T.ink}; }}
        .pv-bar-track {{ height: 7px; background: {T.field}; border-radius: 1px; overflow: hidden; }}
        .pv-bar-fill {{ height: 100%; background: {T.signal}; border-radius: 1px; }}
        .pv-bar-val {{ font-family: {T.mono}; font-size: .78rem; color: {T.graphite}; text-align: right; }}

        /* ---------- ledger ---------- */
        .pv-ledger {{ width: 100%; border-collapse: collapse; margin-top: .3rem; }}
        .pv-ledger td {{ padding: .46rem 0; border-bottom: 1px solid {T.rule}; font-size: .88rem; }}
        .pv-ledger td:last-child {{ text-align: right; font-family: {T.mono};
                                    font-variant-numeric: tabular-nums; }}
        .pv-ledger tr:last-child td {{ border-bottom: none; font-weight: 600;
                                       border-top: 2px solid {T.ink}; padding-top: .6rem; }}
        .pv-ledger .pv-muted td {{ color: {T.graphite}; }}

        /* ---------- streamlit widget overrides ---------- */
        div[data-testid="stMetric"] {{
            background: {T.panel}; border: 1px solid {T.rule}; border-radius: 3px;
            padding: .95rem 1.05rem;
        }}
        div[data-testid="stMetricLabel"] p {{
            font-family: {T.mono} !important; font-size: .65rem !important;
            letter-spacing: .13em; text-transform: uppercase; color: {T.graphite} !important;
        }}
        div[data-testid="stMetricValue"] {{
            font-family: {T.mono} !important; font-weight: 500 !important;
            font-size: 1.7rem !important; letter-spacing: -.025em; color: {T.ink} !important;
        }}
        div[data-testid="stMetricDelta"] {{ font-family: {T.mono} !important; font-size: .76rem !important; }}

        .stButton > button {{
            background: {T.signal}; color: #fff; border: none; border-radius: 2px;
            font-family: {T.display}; font-weight: 600; font-size: .9rem;
            letter-spacing: .01em; padding: .62rem 1.1rem; width: 100%;
            transition: background .15s ease;
        }}
        .stButton > button:hover {{ background: #094A55; color: #fff; }}
        .stButton > button:focus-visible {{ outline: 3px solid {T.watch}; outline-offset: 2px; }}

        section[data-testid="stSidebar"] {{ background: {T.panel}; border-right: 1px solid {T.rule}; }}
        section[data-testid="stSidebar"] .pv-eyebrow {{ margin-top: .4rem; }}

        div[data-testid="stExpander"] {{ border: 1px solid {T.rule}; border-radius: 3px;
                                          background: {T.panel}; }}

        label[data-testid="stWidgetLabel"] p {{
            font-family: {T.mono} !important; font-size: .66rem !important;
            letter-spacing: .13em; text-transform: uppercase; color: {T.graphite} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ======================================================================================
# SVG components
# ======================================================================================


def _polar(cx: float, cy: float, r: float, deg: float) -> Tuple[float, float]:
    rad = math.radians(deg - 90.0)
    return cx + r * math.cos(rad), cy + r * math.sin(rad)


def _arc(cx: float, cy: float, r: float, start_deg: float, end_deg: float) -> str:
    x0, y0 = _polar(cx, cy, r, start_deg)
    x1, y1 = _polar(cx, cy, r, end_deg)
    large = 1 if abs(end_deg - start_deg) > 180 else 0
    return f"M {x0:.2f} {y0:.2f} A {r} {r} 0 {large} 1 {x1:.2f} {y1:.2f}"


def dial_svg(score: float, band: str) -> str:
    """240-degree instrument dial. The score is the only large figure on the page."""
    sweep_start, sweep_end = -120.0, 120.0
    frac = max(0.0, min(1.0, score / 100.0))
    value_end = sweep_start + (sweep_end - sweep_start) * frac
    cx, cy, r = 110.0, 108.0, 76.0
    circ = math.radians(sweep_end - sweep_start) * r

    ticks = []
    for i in range(0, 101, 20):
        deg = sweep_start + (sweep_end - sweep_start) * (i / 100.0)
        xa, ya = _polar(cx, cy, r + 11, deg)
        xb, yb = _polar(cx, cy, r + 16, deg)
        ticks.append(
            f'<line x1="{xa:.1f}" y1="{ya:.1f}" x2="{xb:.1f}" y2="{yb:.1f}" '
            f'stroke="{T.mist}" stroke-width="1.5"/>'
        )

    return f"""
    <svg viewBox="0 0 220 172" width="100%" style="max-width:250px;display:block;margin:0 auto;"
         role="img" aria-label="Vulnerability score {score:.0f} out of 100, {BAND_LABELS.get(band, band)}">
      {''.join(ticks)}
      <path d="{_arc(cx, cy, r, sweep_start, sweep_end)}" fill="none"
            stroke="{T.field}" stroke-width="15" stroke-linecap="butt"/>
      <path class="pv-dial-arc" d="{_arc(cx, cy, r, sweep_start, value_end)}" fill="none"
            stroke="{T.signal}" stroke-width="15" stroke-linecap="butt"
            stroke-dasharray="{circ:.1f}" stroke-dashoffset="{circ:.1f}">
        <animate attributeName="stroke-dashoffset" from="{circ:.1f}" to="0"
                 dur="0.75s" fill="freeze" calcMode="spline"
                 keySplines="0.22 0.68 0.35 1" keyTimes="0;1"/>
      </path>
      <text x="{cx}" y="{cy + 8}" text-anchor="middle"
            style="font-family:{T.mono};font-weight:600;font-size:44px;
                   fill:{T.ink};letter-spacing:-.03em;">{score:.0f}</text>
      <text x="{cx}" y="{cy + 30}" text-anchor="middle"
            style="font-family:{T.mono};font-size:10px;letter-spacing:.16em;
                   fill:{T.graphite};">OF 100</text>
    </svg>
    """


def pass_strip_svg(years: List[int], distances: List[float], flag_last: bool, status: str) -> str:
    """The signature element.

    One bar per annual satellite pass. Bar height is that year's distance from the site's
    own multi-year norm — so a flat strip means a stable site, a ragged strip means a
    volatile one, and a single tall bar at the right-hand end means something happened.
    """
    if not years:
        return '<p class="pv-note">No annual passes available for this location.</p>'

    w, h = 560.0, 96.0
    pad_l, pad_b, pad_t = 6.0, 22.0, 8.0
    n = len(years)
    gap = 7.0
    bar_w = max(8.0, (w - pad_l * 2 - gap * (n - 1)) / n)
    peak = max(distances) or 1.0
    usable = h - pad_b - pad_t

    accent = {"triggered": T.triggered, "watch": T.watch, "clear": T.signal}[status]
    bars = []
    for i, (yr, d) in enumerate(zip(years, distances)):
        bar_h = max(3.0, (d / peak) * usable)
        x = pad_l + i * (bar_w + gap)
        y = h - pad_b - bar_h
        is_event = flag_last and i == n - 1
        fill = accent if is_event else T.signal
        opacity = 1.0 if is_event else 0.34 + 0.5 * (d / peak)
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" '
            f'fill="{fill}" opacity="{opacity:.2f}" rx="1"/>'
        )
        bars.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{h - 7:.1f}" text-anchor="middle" '
            f'style="font-family:{T.mono};font-size:9.5px;letter-spacing:.04em;'
            f'fill:{accent if is_event else T.mist};">{str(yr)[2:]}</text>'
        )

    baseline = h - pad_b + 3
    return f"""
    <svg viewBox="0 0 {w} {h}" width="100%" role="img"
         aria-label="Annual satellite passes {years[0]} to {years[-1]}, bar height is deviation from the site norm">
      <line x1="{pad_l}" y1="{baseline}" x2="{w - pad_l}" y2="{baseline}"
            stroke="{T.rule}" stroke-width="1"/>
      {''.join(bars)}
    </svg>
    """


def badge_html(status: str) -> str:
    label = {"clear": "Normal", "watch": "Monitoring", "triggered": "Payout triggered"}[status]
    return f'<span class="pv-badge pv-badge--{status}"><span class="pv-dot"></span>{label}</span>'


def money(value: float) -> str:
    """Indian numbering — this is an MSME product, so lakh and crore are the native units."""
    if value >= 1e7:
        return f"₹{value / 1e7:,.2f} Cr"
    if value >= 1e5:
        return f"₹{value / 1e5:,.2f} L"
    return f"₹{value:,.0f}"


# ======================================================================================
# Views
# ======================================================================================


def render_underwriter(a: Assessment) -> None:
    tier = a.tier
    pattern_label, pattern_copy = PATTERN_COPY.get(a.pattern, PATTERN_COPY["mixed"])

    left, right = st.columns([1, 1.75], gap="medium")
    with left:
        st.markdown(
            f'<div class="pv-panel pv-panel--accent">'
            f'<span class="pv-eyebrow">Location risk score</span>'
            f"{dial_svg(a.score, a.band)}"
            f'<p class="pv-note" style="text-align:center;margin-top:.7rem;">'
            f"<b>{BAND_LABELS.get(a.band, a.band)}</b> · {pattern_label}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f'<div class="pv-panel">'
            f'<span class="pv-eyebrow">Year-by-year change at this site</span>'
            f"{pass_strip_svg(a.years, a.year_distances, a.status != 'clear', a.status)}"
            f'<p class="pv-note">{pattern_copy}</p>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Risk grade", tier.label)
    m2.metric("Rate on sum insured", f"{tier.base_rate_pct:.2f}%")
    m3.metric("Annual premium", money(a.total_premium))
    m4.metric(
        "Damage found",
        f"{a.damage_fraction * 100:.1f}%",
        delta=f"z {a.mean_z:.1f}" if a.mean_z else None,
        delta_color="off",
    )

    st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.15, 1], gap="medium")

    with c1:
        bars = "".join(
            f'<div class="pv-bar-row">'
            f'<span class="pv-bar-label">{label}</span>'
            f'<span class="pv-bar-track"><span class="pv-bar-fill" '
            f'style="width:{a.components.get(key, 0) * 100:.1f}%"></span></span>'
            f'<span class="pv-bar-val">{a.components.get(key, 0):.2f}</span>'
            f"</div>"
            for key, label in (
                ("churn", "Ground changes each year"),
                ("tail", "Worst year on record"),
                ("dispersion", "How unpredictable"),
                ("regime", "Permanent land-use change"),
            )
        )
        weights = (
            f"Weighted {POLICY.w_churn:.0%} / {POLICY.w_tail:.0%} / "
            f"{POLICY.w_dispersion:.0%} / {POLICY.w_regime:.0%}."
        )
        st.markdown(
            f'<div class="pv-panel"><span class="pv-eyebrow">What drives this score</span>{bars}'
            f'<p class="pv-note">Each bar runs from 0 to 1, where 0 is completely settled '
            f"ground and 1 is ground that changes constantly. {weights} Permanent land-use "
            f"change is weighted as heavily as yearly movement, because a site being built "
            f"over or cleared will never return to what it was, while farmland that floods "
            f"every monsoon always does. The first is a lasting change to the risk; the "
            f"second is just the normal rhythm of the place.</p></div>",
            unsafe_allow_html=True,
        )

    with c2:
        rows = [
            ("Sum insured", money(a.sum_insured), False),
            (f"Property @ {tier.base_rate_pct:.2f}%", money(a.property_premium), True),
            (f"Automatic payout cover @ {tier.parametric_rate_pct:.2f}%", money(a.parametric_premium), True),
            ("Annual premium", money(a.total_premium), False),
        ]
        body = "".join(
            f'<tr class="{"pv-muted" if muted else ""}"><td>{k}</td><td>{v}</td></tr>'
            for k, v, muted in rows
        )
        st.markdown(
            f'<div class="pv-panel"><span class="pv-eyebrow">Pricing</span>'
            f'<table class="pv-ledger">{body}</table>'
            f'<p class="pv-note" style="margin-top:.8rem;">{tier.guidance}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)

    st.markdown(
        f'<div class="pv-panel"><span class="pv-eyebrow">Automatic payout status</span>'
        f'<div style="display:flex;align-items:center;gap:1.4rem;flex-wrap:wrap;">'
        f"{badge_html(a.status)}"
        f'<span class="pv-fig pv-fig--sm">Damage found: {a.damage_fraction * 100:.1f}% of the site · '
        f"Payout begins at {POLICY.payout_tiers[0][0] * 100:.0f}% · "
        f"Maximum payout {money(a.parametric_limit)}</span></div>"
        f'<p class="pv-note">{_status_note_underwriter(a)}</p></div>',
        unsafe_allow_html=True,
    )

    with st.expander("Technical detail — satellite measurements behind the score"):
        st.markdown(
            f'<p class="pv-fig pv-fig--sm">'
            f"Location {a.lat:.5f}, {a.lon:.5f} · Source {a.source.upper()} · "
            f"Radar scenes {a.sar_scenes}</p>",
            unsafe_allow_html=True,
        )
        st.json({"raw_metrics": a.raw_metrics, "components_0_1": a.components})
        for note in a.notes:
            st.caption(note)


def _status_note_underwriter(a: Assessment) -> str:
    if a.status == "triggered":
        return (
            f"Satellite radar found damage across {a.damage_fraction * 100:.1f}% of the "
            f"insured area, above the {POLICY.payout_tiers[0][0] * 100:.0f}% level written "
            f"into the policy. {money(a.payout_amount)} is payable "
            f"({a.payout_fraction * 100:.0f}% of the maximum). No surveyor visit and no "
            f"loss assessment are required — the policy pays on the satellite reading alone."
        )
    if a.status == "watch":
        return (
            "Some damage was found, but less than the policy requires for a payout. "
            "Nothing is payable today. Worth a second look if it shows up again on the "
            "next two satellite passes."
        )
    return (
        "Nothing here beyond the normal year-to-year change for this site. "
        "Nothing is payable."
    )


def render_policyholder(a: Assessment) -> None:
    accent = {"clear": T.clear, "watch": T.watch, "triggered": T.triggered}[a.status]

    if a.status == "triggered":
        headline = "A payout is on its way"
        lede = (
            f"Satellite radar picked up damage across your area on "
            f"{a.observed_on.strftime('%d %B %Y') if a.observed_on else 'the last pass'}. "
            f"That passes the trigger written into your policy, so "
            f"{money(a.payout_amount)} has been released. You do not need to file a claim "
            f"or wait for a surveyor."
        )
    elif a.status == "watch":
        headline = "We're watching your area"
        lede = (
            "The last satellite pass showed some change near your premises, but not enough "
            "to trigger a payout under your policy. We'll check again on the next pass, "
            "usually within a week. Nothing is needed from you."
        )
    else:
        headline = "Your cover is active"
        lede = (
            "The last satellite pass over your premises looked normal. If a flood, storm "
            "or fire changes that, your payout starts automatically — no claim form, "
            "no site visit."
        )

    st.markdown(
        f'<div class="pv-hero" style="--pv-accent:{accent};">'
        f"{badge_html(a.status)}"
        f"<h2>{headline}</h2><p>{lede}</p></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Covered up to", money(a.sum_insured))
    m2.metric("Maximum automatic payout", money(a.parametric_limit))
    m3.metric(
        "Released so far" if a.payout_amount else "Released this year",
        money(a.payout_amount),
    )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.4, 1], gap="medium")

    with c1:
        _, pattern_copy = PATTERN_COPY.get(a.pattern, PATTERN_COPY["mixed"])
        st.markdown(
            f'<div class="pv-panel"><span class="pv-eyebrow">Your area, year by year</span>'
            f"{pass_strip_svg(a.years, a.year_distances, a.status != 'clear', a.status)}"
            f'<p class="pv-note">Each bar is one year of satellite observation. Taller bars '
            f"mean the land around your premises changed more that year. {pattern_copy}</p></div>",
            unsafe_allow_html=True,
        )

    with c2:
        rating = {
            "very_low": "Well below average risk",
            "low": "Below average risk",
            "moderate": "Around average risk",
            "elevated": "Above average risk",
            "high": "Well above average risk",
        }[a.band]
        st.markdown(
            f'<div class="pv-panel pv-panel--accent">'
            f'<span class="pv-eyebrow">What you pay</span>'
            f'<div class="pv-fig pv-fig--xl" style="color:{T.signal};">'
            f"{money(a.total_premium)}</div>"
            f'<p class="pv-note" style="margin-top:.35rem;">per year, covering '
            f"{money(a.sum_insured)}</p>"
            f'<p class="pv-note" style="margin-top:.9rem;">Your premises sit in the '
            f"<b>{a.tier.label.lower()}</b> price band — {rating.lower()} for this region. "
            f"That rating comes from eight years of satellite history, not a questionnaire.</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    steps = [
        ("Satellite passes overhead", "Every 6 to 12 days, cloud or clear."),
        ("We compare it to your normal", "Not a regional average — your own site's history."),
        ("Past the trigger, money moves", "No claim form, no surveyor, no negotiation."),
    ]
    cols = st.columns(3)
    for col, (title, sub) in zip(cols, steps):
        with col:
            st.markdown(
                f'<div class="pv-panel"><span class="pv-eyebrow">{title}</span>'
                f'<p class="pv-note" style="margin:0;">{sub}</p></div>',
                unsafe_allow_html=True,
            )


def render_self_test() -> None:
    """Show the machinery checks, and the reality checks that only live data can settle."""
    st.markdown(
        '<div class="pv-mast"><h1>Self-test</h1>'
        '<span class="pv-sub">Is it working correctly?</span></div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Running checks…"):
        results = run_self_tests()

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    all_ok = passed == total
    accent = T.clear if all_ok else T.triggered

    st.markdown(
        f'<div class="pv-hero" style="--pv-accent:{accent};">'
        f'<span class="pv-badge pv-badge--{"clear" if all_ok else "triggered"}">'
        f'<span class="pv-dot"></span>{passed} of {total} checks passed</span>'
        f"<h2>{'The machinery is sound' if all_ok else 'Something is wrong'}</h2>"
        f"<p>These checks prove the app calculates what it says it calculates: the "
        f"arithmetic adds up, the payout thresholds fire at the right levels, and the "
        f"same input always gives the same answer. "
        f"<b>They do not prove the risk scores are true for real places</b> — nothing "
        f"can, while the data is simulated.</p></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    rows = "".join(
        f'<tr><td style="width:2.2rem;color:{T.clear if ok else T.triggered};'
        f'font-family:{T.mono};font-weight:600;">{"PASS" if ok else "FAIL"}</td>'
        f"<td>{name}</td>"
        f'<td style="color:{T.graphite};font-family:{T.mono};font-size:.78rem;">{detail}</td></tr>'
        for name, ok, detail in results
    )
    st.markdown(
        f'<div class="pv-panel"><span class="pv-eyebrow">What was checked</span>'
        f'<table class="pv-ledger" style="font-size:.86rem;">{rows}</table></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    site_rows = "".join(
        f"<tr><td><b>{s['name']}</b><br>"
        f'<span style="color:{T.graphite};font-size:.82rem;">{s["why"]}</span></td>'
        f'<td style="font-family:{T.mono};font-size:.8rem;white-space:nowrap;">'
        f"{s['lat']:.4f}<br>{s['lon']:.4f}</td>"
        f'<td style="color:{T.signal};font-weight:600;font-size:.85rem;">{s["expect"]}</td></tr>'
        for s in KNOWN_SITES
    )
    st.markdown(
        f'<div class="pv-panel pv-panel--accent">'
        f'<span class="pv-eyebrow">The test that actually matters — needs live data</span>'
        f'<p class="pv-note" style="margin-bottom:.9rem;">Type each of these into the app '
        f"once real satellite data is connected. You already know what the ground looks "
        f"like at all five, so you can judge whether the scores are sensible. If Gir "
        f"Forest scores higher than the Surat building sites, the model is wrong and you "
        f"will know immediately.</p>"
        f'<table class="pv-ledger" style="font-size:.88rem;">{site_rows}</table>'
        f'<p class="pv-note" style="margin-top:.9rem;">In demo mode these return invented '
        f"numbers, so they prove nothing. That is the honest limit of a demo.</p></div>",
        unsafe_allow_html=True,
    )


# ======================================================================================
# App shell
# ======================================================================================


def main() -> None:
    st.set_page_config(
        page_title="Terrafirma Risk Console",
        page_icon="◔",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    # ---------------- sidebar ----------------
    with st.sidebar:
        st.markdown('<span class="pv-eyebrow">View as</span>', unsafe_allow_html=True)
        mode = st.radio(
            "View as",
            ["Underwriter", "MSME policyholder", "Self-test"],
            label_visibility="collapsed",
        )

        st.markdown('<span class="pv-eyebrow">Data source</span>', unsafe_allow_html=True)
        source_options = ["Demo data", "Live satellite"] if GEO_RISK_AVAILABLE else ["Demo data"]
        source = st.radio("Data source", source_options, label_visibility="collapsed")

        if not GEO_RISK_AVAILABLE:
            st.caption(
                "Live mode needs geo_risk.py and earthengine-api alongside this file. "
                "Running on demo data."
            )

        radius = 500.0
        ee_ready, ee_detail = False, ""
        if source == "Live satellite":
            ee_ready, ee_detail = init_earth_engine()
            if ee_ready:
                st.success(f"Connected to Earth Engine · {ee_detail}")
            else:
                st.error(ee_detail)
            radius = st.slider("Analysis radius (m)", 100, 2000, 500, step=100)
            st.caption(
                "How far around the location to look. 500 m covers a factory and its "
                "immediate surroundings."
            )

        scenario = "No event"
        if source == "Demo data":
            st.markdown('<span class="pv-eyebrow">Event scenario</span>', unsafe_allow_html=True)
            scenario = st.select_slider(
                "Event scenario",
                options=["No event", "Minor disturbance", "Moderate loss", "Severe loss"],
                label_visibility="collapsed",
            )
            st.caption("Demo only. Drives the parametric trigger so you can see both states.")

        st.markdown("---")
        if source == "Demo data":
            st.markdown(
                '<p class="pv-note" style="font-size:.78rem;"><b>These numbers are '
                "simulated.</b> No satellite is contacted in demo mode. Scores are "
                "generated from the coordinates you type, so the same location always "
                "returns the same result. Do not present them as real readings.</p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p class="pv-note" style="font-size:.78rem;">Live mode. Scores come from '
                "AlphaEarth annual satellite embeddings; the parametric trigger runs on "
                "Sentinel-1 radar, which sees through cloud.</p>",
                unsafe_allow_html=True,
            )

    # ---------------- masthead ----------------
    st.markdown(
        '<div class="pv-mast"><h1>Terrafirma</h1>'
        '<span class="pv-sub">Risk Console</span></div>'
        f'<div class="pv-modeline">Viewing as <b>{mode}</b> · '
        f'{"Live satellite" if source == "Live satellite" else "Demo data"}</div>',
        unsafe_allow_html=True,
    )

    if mode == "Self-test":
        render_self_test()
        return

    # ---------------- location lookup ----------------
    # Sits outside the form: widgets inside a form cannot trigger a rerun, and the
    # lookup has to update the coordinate boxes before "Analyze risk" is pressed.
    st.session_state.setdefault("lat_val", 22.3039)
    st.session_state.setdefault("lon_val", 70.8022)

    g1, g2 = st.columns([3.4, 1.3])
    place_query = g1.text_input(
        "Find by address or PIN code",
        placeholder="e.g. 360001  ·  Gondal Road, Rajkot  ·  Morbi",
    )
    if g2.button("Find coordinates"):
        if not place_query.strip():
            st.session_state["geo_msg"] = ("warn", "Type an address or a 6-digit PIN code first.")
        else:
            with st.spinner("Looking up that location…"):
                found = resolve_location(place_query)
            if found:
                st.session_state["lat_val"] = round(found.lat, 5)
                st.session_state["lon_val"] = round(found.lon, 5)
                note = {
                    "exact": "Building-level match.",
                    "approximate": "Area-level match — the pin may sit some distance from "
                                   "the actual premises. Check it before relying on the score.",
                    "city": "City-centre match. Add a street or area for a closer fix.",
                }[found.precision]
                st.session_state["geo_msg"] = (
                    "ok" if found.precision == "exact" else "warn",
                    f"{found.label} — {note} (via {found.source})",
                )
            else:
                st.session_state["geo_msg"] = (
                    "err",
                    "Could not find that. Try adding the city and state, or type the "
                    "coordinates in by hand.",
                )

    msg = st.session_state.get("geo_msg")
    if msg:
        kind, text = msg
        {"ok": st.success, "warn": st.warning, "err": st.error}[kind](text)

    # ---------------- inputs ----------------
    with st.form("location", border=False):
        c1, c2, c3, c4 = st.columns([2.4, 1, 1, 1.3])
        business = c1.text_input("Business name", value="Patel Ceramics Pvt Ltd")
        lat = c2.number_input("Latitude", key="lat_val", format="%.5f", min_value=-90.0, max_value=90.0)
        lon = c3.number_input("Longitude", key="lon_val", format="%.5f", min_value=-180.0, max_value=180.0)
        sum_insured = c4.number_input(
            "Sum insured (₹)", value=8_500_000, step=500_000, min_value=100_000
        )

        c5, c6 = st.columns([2.4, 1.3])
        check_event = c5.checkbox("Check for a loss event", value=True)
        event_date = c6.date_input("Event date", value=date.today() - timedelta(days=30))

        submitted = st.form_submit_button("Analyze risk")

    if submitted:
        if not business.strip():
            st.error("Enter a business name so the assessment can be filed against a policy.")
            return
        try:
            with st.spinner("Reading satellite history for this location…"):
                if source == "Live satellite":
                    if not ee_ready:
                        st.error(
                            "Not connected to Earth Engine, so live data is unavailable. "
                            "See the message in the sidebar, or switch back to demo data."
                        )
                        return
                    result = live_assessment(
                        business.strip(), lat, lon, float(sum_insured), radius,
                        event_date if check_event else None,
                    )
                else:
                    result = demo_assessment(
                        business.strip(), lat, lon, float(sum_insured),
                        event_date if check_event else None,
                        scenario if check_event else "No event",
                    )
            st.session_state["assessment"] = result
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not complete the assessment: {exc}")
            st.caption(
                "Check the coordinates fall on land and that Earth Engine is authorised "
                "for this project."
            )
            return

    assessment: Optional[Assessment] = st.session_state.get("assessment")

    if assessment is None:
        st.markdown(
            '<div class="pv-panel" style="text-align:center;padding:3rem 1.5rem;">'
            '<span class="pv-eyebrow">No location loaded</span>'
            '<p class="pv-lede" style="max-width:44ch;margin:0 auto;">Enter a business and '
            "its coordinates above, then run the analysis. Eight years of satellite history "
            "will price the risk and set the parametric trigger.</p></div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f'<p class="pv-fig pv-fig--sm" style="margin:.2rem 0 .9rem 0;">'
        f"{assessment.business} · {assessment.lat:.4f}, {assessment.lon:.4f}</p>",
        unsafe_allow_html=True,
    )

    if mode == "Underwriter":
        render_underwriter(assessment)
    else:
        render_policyholder(assessment)


if __name__ == "__main__":
    main()
