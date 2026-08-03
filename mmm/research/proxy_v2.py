"""
Calibrated AI-likeness proxy v2 — SH-family features + piecewise-linear
calibration against real SubmitHub checker results.

Anchors (measured with the SubmitHub AI Song Checker, researcher's
reports of 2026-08-03):

    file             spectral (Pure AI)   temporal (Pure AI)
    06_fast          87%                  92%
    03_paranoid      86%                  32%
    05_stealth_plus  84%                  44%

Calibration is a piecewise-linear interpolation of raw machine-likeness
mean -> checker sub-score through the anchor points (monotone, clamped at
the edges).  Three anchors are enough to see that the checker's response
is *nonlinear and family-specific*:

- spectral: essentially flat across all processing (84-87) even though
  raw spectral features spread wide (0.54-0.78) — the real spectral
  classifier keys on something our hand-crafted features do not capture.
- temporal: steep near the unprocessed track (92), dropping to ~32-44 for
  any humanization-style processing — matching the raw ordering
  (paranoid < stealth_plus < fast).

The calibration reproduces the anchors exactly and interpolates between
them; outside the anchor range it clamps to the edge value.  It is NOT
validated on unseen files yet — more anchors (e.g. the single-stage
ablation pack) will improve it.

Directions: each term maps to machine-likeness in [0,1] where 1 means
"more machine-like".  Reference ranges are set from the observed values
on the calibration files and from the documented feature semantics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .shlabs import extract_sh_features
from .features import extract_features

# Calibration parameters are persisted here by calibrate() so that every
# consumer (benchmark, ablation) applies the same fitted maps.
_CALIBRATION_PATH = Path(__file__).parent / "calibration.json"


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _term(features: dict, key: str, machine_at_high: bool, lo: float, hi: float) -> float | None:
    val = features.get(key)
    if val is None or not np.isfinite(val):
        return None
    x = _clip01((val - lo) / (hi - lo))
    return x if machine_at_high else 1.0 - x


# (feature, machine_at_high, (lo, hi), weight)
_SPECTRAL_TERMS = [
    # Low rolloff => narrowband => machine-like (Suno mixes are band-limited).
    ("sh_rolloff85_mean", False, (6000.0, 12000.0), 1.0),
    # Stationary spectrum (low flatness std) => machine-like.
    ("sh_flatness_std", False, (0.002, 0.015), 1.0),
    # Low MFCC dynamics => machine-like.
    ("sh_mfcc_std_1", False, (15.0, 80.0), 1.0),
    ("sh_mfcc_std_2", False, (12.0, 70.0), 0.7),
    # Concentrated DCT texture => machine-like (spectrogram "picture").
    ("sh_texture_gini", True, (0.50, 0.62), 1.0),
    # Low spectral-entropy variation => machine-like.
    ("spectral_entropy_std", False, (0.05, 0.35), 0.7),
    # Harmonic complexity (fewer distinct partials) => machine-like.
    ("sh_harmonic_complexity", False, (25.0, 60.0), 0.5),
]

_TEMPORAL_TERMS = [
    # Data-selected on 4 real-checker anchors (2026-08-03): only features
    # whose machine-likeness term correlates POSITIVELY (Spearman > 0.5)
    # with the real temporal score are kept.  beat_cv / sh_rms_kurt /
    # rms_skew all had term-rho = +0.80; the other candidates
    # (onset_regularity, sh_zcr_std, sh_section_corr_60, sh_tempo_chunk_std,
    # sh_rms_std, tempo-chunk mean) were inverted or noise on this corpus.
    # Irregular beat (high CV) => human-like.
    ("beat_cv", False, (0.005, 0.15), 1.0),
    # RMS kurtosis: high (peaky dynamics) => human-like.
    ("sh_rms_kurt", False, (0.0, 5.0), 1.0),
    # RMS skew: high => human-like.
    ("rms_skew", False, (0.1, 2.5), 1.0),
]

# Piecewise-linear mapping tables: raw (0-1) -> checker sub-score (0-100).
# Built by calibrate(); loaded from calibration.json by _load_calibration().
_SPEC_RAWS: list[float] = []
_SPEC_REALS: list[float] = []
_TEMP_RAWS: list[float] = []
_TEMP_REALS: list[float] = []

ANCHORS = {
    "06_fast": {"spectral": 87.0, "temporal": 92.0, "note": "unprocessed (fast = tag strip only)"},
    "03_paranoid": {"spectral": 86.0, "temporal": 32.0, "note": "paranoid variant"},
    "05_stealth_plus": {"spectral": 84.0, "temporal": 44.0, "note": "stealth-plus variant"},
    "ablation_tempo_drift": {
        "spectral": 90.0,
        "temporal": 34.0,
        "note": "single-stage tempo drift: real temporal dropped 92->34 but NO "
        "hand-crafted temporal feature captures it (beat CV, RMS dynamics, "
        "tempo stability, beat-phase drift all fail to rank it). Excluded "
        "from the temporal fit; the fingerprint DeltaMatch (~0.225) is the "
        "best in-tool predictor for drift-like stages.",
        "exclude_temporal": True,
    },
}


def raw_subscore(features: dict, kind: str) -> float:
    """Weighted mean machine-likeness for a family, ignoring missing terms."""
    terms = _SPECTRAL_TERMS if kind == "spectral" else _TEMPORAL_TERMS
    vals = []
    weights = []
    for key, high, (lo, hi), w in terms:
        t = _term(features, key, high, lo, hi)
        if t is not None:
            vals.append(t)
            weights.append(w)
    if not vals:
        return 0.5
    return float(np.average(vals, weights=weights))


def _sorted_pairs(pairs: list[tuple[float, float]]) -> tuple[list[float], list[float]]:
    pairs = sorted(pairs, key=lambda p: p[0])
    return [p[0] for p in pairs], [p[1] for p in pairs]


def calibrate(calib_features: dict) -> dict:
    """
    Build piecewise-linear anchor maps from raw sub-scores to the anchor
    checker scores.  Persists to calibration.json.  Returns
    {"spectral": {"raw": [...], "real": [...]}, "temporal": {...}}.
    """
    global _SPEC_RAWS, _SPEC_REALS, _TEMP_RAWS, _TEMP_REALS

    spec_pairs: list[tuple[float, float]] = []
    temp_pairs: list[tuple[float, float]] = []
    used_anchors: dict[str, dict] = {}
    for name, targets in ANCHORS.items():
        f = calib_features.get(name)
        if f is None:
            continue
        spec_pairs.append((raw_subscore(f, "spectral"), float(targets["spectral"])))
        if targets.get("exclude_temporal"):
            continue
        temp_pairs.append((raw_subscore(f, "temporal"), float(targets["temporal"])))
        used_anchors[name] = targets

    _SPEC_RAWS, _SPEC_REALS = _sorted_pairs(spec_pairs)
    _TEMP_RAWS, _TEMP_REALS = _sorted_pairs(temp_pairs)

    payload = {
        "spectral": {"raw": _SPEC_RAWS, "real": _SPEC_REALS},
        "temporal": {"raw": _TEMP_RAWS, "real": _TEMP_REALS},
        "anchors": used_anchors,
        "note": "Piecewise-linear interpolation through the reported SubmitHub "
        "checker results (fast 87/92, paranoid 86/32, stealth_plus 84/44). "
        "Interpolates between anchors; clamps outside. Refit as more anchor "
        "points become available.",
    }
    _CALIBRATION_PATH.write_text(json.dumps(payload, indent=2))
    return payload


def calibrate_from_raw(points: dict) -> dict:
    """
    Build piecewise-linear anchor maps directly from raw sub-scores
    (0-100 scale) + real checker sub-scores.  Accepts ablation-style
    anchors where only the raw proxy scores are known:

        points = {"ablation_tempo_drift": {
            "raw_spec": 63.0, "raw_temp": 51.0, "spec": 90.0, "temp": 34.0}}

    Persists to calibration.json (same format as calibrate()).
    """
    global _SPEC_RAWS, _SPEC_REALS, _TEMP_RAWS, _TEMP_REALS

    spec_pairs = [(float(p["raw_spec"]) / 100.0, float(p["spec"])) for p in points.values()]
    temp_pairs = [(float(p["raw_temp"]) / 100.0, float(p["temp"])) for p in points.values()]

    _SPEC_RAWS, _SPEC_REALS = _sorted_pairs(spec_pairs)
    _TEMP_RAWS, _TEMP_REALS = _sorted_pairs(temp_pairs)

    payload = {
        "spectral": {"raw": _SPEC_RAWS, "real": _SPEC_REALS},
        "temporal": {"raw": _TEMP_RAWS, "real": _TEMP_REALS},
        "anchors": points,
        "note": "Piecewise-linear interpolation through real SubmitHub checker "
        "results. Interpolates between anchors; clamps outside. Refit as more "
        "anchor points become available.",
    }
    _CALIBRATION_PATH.write_text(json.dumps(payload, indent=2))
    return payload


def _load_calibration() -> None:
    """Load persisted calibration params if present (idempotent)."""
    global _SPEC_RAWS, _SPEC_REALS, _TEMP_RAWS, _TEMP_REALS
    if _CALIBRATION_PATH.exists() and not _SPEC_RAWS:
        try:
            data = json.loads(_CALIBRATION_PATH.read_text())
            _SPEC_RAWS = data["spectral"]["raw"]
            _SPEC_REALS = data["spectral"]["real"]
            _TEMP_RAWS = data["temporal"]["raw"]
            _TEMP_REALS = data["temporal"]["real"]
        except Exception:
            pass


def _apply(raw: float, raws: list[float], reals: list[float]) -> float:
    if not raws:
        return round(100.0 * raw, 2)
    xp = np.asarray(raws, dtype=np.float64)
    fp = np.asarray(reals, dtype=np.float64)
    return round(float(np.interp(raw, xp, fp)), 2)


def score_ai_likeness(features: dict) -> dict:
    """
    Calibrated proxy score.  Expects a combined feature dict (extract_
    features + extract_sh_features merged).  Returns overall/spectral/
    temporal 0-100 plus raw (uncalibrated) sub-scores and per-term values.
    """
    _load_calibration()
    raw_spec = raw_subscore(features, "spectral")
    raw_temp = raw_subscore(features, "temporal")
    # Spectral: pinned-band model.  All hand-crafted spectral features are
    # ANTI-correlated with the real checker's spectral score (Spearman
    # -0.6..-1.0 across 4 anchors) while the real score stays in a narrow
    # band (84-90) across drastically different processing.  The honest
    # model is: report the observed band (anchor mean +/- range), do not
    # pretend a fit.  raw_spectral is still reported for direction.
    spec = round(float(np.mean(_SPEC_REALS)), 1) if _SPEC_REALS else round(100.0 * raw_spec, 2)
    temp = _apply(raw_temp, _TEMP_RAWS, _TEMP_REALS)
    return {
        "overall": round(0.5 * spec + 0.5 * temp, 2),
        "spectral": spec,
        "temporal": temp,
        "raw_spectral": round(100.0 * raw_spec, 2),
        "raw_temporal": round(100.0 * raw_temp, 2),
        "spectral_note": "pinned-band: real spectral stayed 84-90 across all "
        "processing; hand-crafted spectral features are anti-correlated with "
        "the real spectral score on the 4 anchors, so no fit is attempted.",
        "terms": {
            key: _term(features, key, high, lo, hi)
            for key, high, (lo, hi), _ in _SPECTRAL_TERMS + _TEMPORAL_TERMS
        },
    }


def extract_all(path_or_audio, sr: int | None = None) -> dict:
    """Extract combined features from a path or (audio, sr)."""
    if isinstance(path_or_audio, str):
        import librosa

        audio, sr = librosa.load(path_or_audio, sr=None, mono=True)
    else:
        audio, sr = path_or_audio, sr
    f = extract_features(audio, sr)
    f.update(extract_sh_features(audio, sr))
    return f


def main(argv=None) -> int:
    """
    CLI: refit the calibration from a features JSON + an anchors JSON, and
    print calibrated scores for every file in the features file.

        python -m mmm.research.proxy_v2 \
            --features bench_results/song1/features_all.json \
            --anchors anchors.json

    anchors.json format: {"06_fast": {"spectral": 87, "temporal": 92}, ...}
    """
    ap = argparse.ArgumentParser(prog="python -m mmm.research.proxy_v2")
    ap.add_argument("--features", required=True, help="features JSON (extract_all output)")
    ap.add_argument(
        "--anchors",
        default=None,
        help="anchors JSON {name: {spectral, temporal}}; default: module ANCHORS",
    )
    args = ap.parse_args(argv)

    with open(args.features, encoding="utf-8") as fh:
        feats = json.load(fh)
    for name in feats:
        for k, v in feats[name].items():
            if isinstance(v, str):
                feats[name][k] = None if v == "None" else float(v)

    if args.anchors:
        with open(args.anchors, encoding="utf-8") as fh:
            custom = json.load(fh)
        for name in list(ANCHORS):
            ANCHORS.pop(name, None)
        for name, targets in custom.items():
            ANCHORS[name] = {"spectral": float(targets["spectral"]),
                             "temporal": float(targets["temporal"])}

    cal = calibrate(feats)
    print("calibration saved to", _CALIBRATION_PATH)
    print("spectral map:", cal["spectral"])
    print("temporal map:", cal["temporal"])
    print("\ncalibrated scores:")
    for name, f in feats.items():
        s = score_ai_likeness(f)
        print(
            f"  {name:18s} overall={s['overall']:6.1f}  spectral={s['spectral']:6.1f}  "
            f"temporal={s['temporal']:6.1f}  (raw {s['raw_spectral']:4.0f}/{s['raw_temporal']:4.0f})"
        )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
