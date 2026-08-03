"""
AI-likeness proxy score (surrogate, interpretable, directional).

Computes a 0-100 "machine-likeness" score from the features produced by
``mmm.research.features``.  The score is an *interpretable surrogate*:

* It is built from published, well-documented *directions* of machine-made
  music statistics (extremely regular timing, low dynamic variation,
  spectral stationarity, low spectral entropy variance, tonal purity).
* It is NOT the SubmitHub AI Song Checker model, and its absolute values
  are NOT comparable to the live checker's output.
* It is intended to support relative before/after measurement (ΔAI proxy)
  inside the sandbox, until the sponsors' evaluation libraries are
  provided.

Each contributing feature is mapped through a fixed, documented reference
range into [0, 1] where 1 means "machine-like".  The spectral and temporal
sub-scores are the means of their families; the overall score is the mean
of the two sub-scores, scaled to 0-100.  Missing features (None) are
excluded from the family mean so short/simple signals do not silently
collapse the score.
"""

from __future__ import annotations

import numpy as np


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


# Each entry: (feature key, machine_like_at_high, reference range)
# The reference ranges are heuristics chosen so that "sterile" synthetic
# material (perfect quantization, constant velocity, static spectra) lands
# near 1 and humanized material lands near 0.  They are documented here on
# purpose; tune them if the calibration sanity check in the report shows
# the surrogate is not separating the synthetic control classes.
_SPECTRAL_TERMS = [
    # Low variance over time => machine-like (stationary spectrum).
    ("centroid_std", False, (60.0, 900.0)),
    ("rolloff95_std", False, (200.0, 2500.0)),
    ("flatness_std", False, (0.03, 0.25)),
    ("spectral_entropy_std", False, (0.03, 0.30)),
    # High peakiness => clean synthetic tones => machine-like.
    ("peakiness_mean", True, (6.0, 60.0)),
    # High-band energy stability => machine-like (HF content static).
    ("hf_energy_ratio_std", False, (0.01, 0.12)),
    # MFCC dynamics: more frame-to-frame variation => human-like.
    ("mfcc_std_1", False, (1.5, 12.0)),
    ("mfcc_std_2", False, (1.5, 12.0)),
    ("mfcc_std_3", False, (1.5, 10.0)),
]

_TEMPORAL_TERMS = [
    # Timing regularity: CV of inter-onset intervals.  Published human
    # performance timing deviation is roughly 0.5-25% CV (tight studio
    # playing ~1-3%, expressive/loose playing 10-25%); grid-quantized
    # machine material is <0.5%.  Reference range: (0.005, 0.25).
    ("onset_regularity", False, (0.005, 0.25)),
    ("beat_cv", False, (0.002, 0.10)),
    # Dynamics: high RMS kurtosis/skew => human expressiveness.
    ("rms_kurtosis", False, (0.5, 8.0)),
    ("rms_skew", False, (0.15, 2.5)),
    # Envelope modulation: low flatness => regular pumping => machine-like.
    ("mod_flatness", False, (0.02, 0.5)),
    # ZCR stability.
    ("zcr_std", False, (0.005, 0.05)),
]


def _term_value(features: dict, key: str, machine_at_high: bool, lo: float, hi: float) -> float | None:
    val = features.get(key)
    if val is None or not np.isfinite(val):
        return None
    x = _clip01((val - lo) / (hi - lo))
    return x if machine_at_high else 1.0 - x


def score_ai_likeness(features: dict) -> dict:
    """
    Compute the surrogate AI-likeness score.

    Returns:
        {
          "overall":  0-100 float,
          "spectral": 0-100 float,
          "temporal": 0-100 float,
          "terms":    {feature_key: normalized_machine_likeness (0..1)},
        }
    """
    spec_terms = [
        _term_value(features, key, high, lo, hi) for key, high, (lo, hi) in _SPECTRAL_TERMS
    ]
    temp_terms = [
        _term_value(features, key, high, lo, hi) for key, high, (lo, hi) in _TEMPORAL_TERMS
    ]

    spec_avail = [t for t in spec_terms if t is not None]
    temp_avail = [t for t in temp_terms if t is not None]

    spectral = float(np.mean(spec_avail)) if spec_avail else 0.5
    temporal = float(np.mean(temp_avail)) if temp_avail else 0.5
    overall = 0.5 * spectral + 0.5 * temporal

    terms: dict = {}
    for (key, _, _), t in zip(_SPECTRAL_TERMS + _TEMPORAL_TERMS, spec_terms + temp_terms):
        terms[key] = t

    return {
        "overall": round(100.0 * overall, 2),
        "spectral": round(100.0 * spectral, 2),
        "temporal": round(100.0 * temporal, 2),
        "terms": terms,
    }
