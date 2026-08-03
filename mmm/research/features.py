"""
Feature extraction for the offline AI-likeness proxy.

Extracts a flat dictionary of scalar audio features from a mono signal.
The feature families mirror the *kinds* of statistics that machine-made
music detectors (e.g. SHLabs-style spectral/temporal classifiers) are
documented to use: spectral statistics, spectral-band energy layout,
tonality/peakiness, temporal regularity (onset/beat stability), dynamics
(RMS distribution), modulation spectrum, and MFCC distribution.

These are NOT the SubmitHub model's features — the feature set is a
transparent, documented surrogate used for directional research only.
"""

from __future__ import annotations

import numpy as np
import librosa

# Default STFT parameters used consistently across extraction.
N_FFT = 2048
HOP = 512


def _safe_stat(values: np.ndarray) -> float | None:
    """Mean of a frame-level feature series, or None if empty/non-finite."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.mean(values))


def _safe_std(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.std(values))


def _spectral_entropy(frame_mag: np.ndarray) -> float:
    """Normalized Shannon entropy of a magnitude spectrum frame (0..1)."""
    p = frame_mag / (np.sum(frame_mag) + 1e-12)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)) / np.log(p.size))


def extract_features(audio: np.ndarray, sr: int) -> dict:
    """
    Extract a flat feature dictionary from a mono float array.

    Returns dict[str, float | None].  None means "not computable for this
    signal" (e.g. too short, no onsets detected) — callers should treat
    None as a neutral/missing value.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.size < N_FFT:
        raise ValueError(f"audio too short for feature extraction ({audio.size} samples)")

    audio = audio / (np.max(np.abs(audio)) + 1e-12)

    feats: dict[str, float | None] = {}

    # ---------------------------------------------------------------- spectral
    S = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP))  # (bins, frames)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    power = S ** 2
    total = power.sum(axis=0) + 1e-12
    n_frames = S.shape[1]

    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    rolloff85 = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0]
    rolloff95 = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.95)[0]
    flatness = librosa.feature.spectral_flatness(S=S)[0]

    feats["centroid_mean"] = _safe_stat(centroid)
    feats["centroid_std"] = _safe_std(centroid)
    feats["bandwidth_mean"] = _safe_stat(bandwidth)
    feats["bandwidth_std"] = _safe_std(bandwidth)
    feats["rolloff85_mean"] = _safe_stat(rolloff85)
    feats["rolloff85_std"] = _safe_std(rolloff85)
    feats["rolloff95_mean"] = _safe_stat(rolloff95)
    feats["rolloff95_std"] = _safe_std(rolloff95)
    feats["flatness_mean"] = _safe_stat(flatness)
    feats["flatness_std"] = _safe_std(flatness)

    # Frame-level spectral entropy + peakiness (tonality).
    entropies = np.array([_spectral_entropy(S[:, t]) for t in range(n_frames)])
    with np.errstate(divide="ignore", invalid="ignore"):
        peakiness = S.max(axis=0) / (S.mean(axis=0) + 1e-12)
    feats["spectral_entropy_mean"] = _safe_stat(entropies)
    feats["spectral_entropy_std"] = _safe_std(entropies)
    feats["peakiness_mean"] = _safe_stat(peakiness)
    feats["peakiness_std"] = _safe_std(peakiness)

    # Band energy layout: 0-500, 500-2k, 2k-8k, 8k+ (bounded by Nyquist).
    band_edges = [0.0, 500.0, 2000.0, 8000.0, sr / 2.0 + 1.0]
    for i in range(len(band_edges) - 1):
        lo, hi = band_edges[i], band_edges[i + 1]
        if hi <= lo:
            continue
        mask = (freqs >= lo) & (freqs < hi)
        if not mask.any():
            continue
        frac = power[mask, :].sum(axis=0) / total
        feats[f"band_{int(lo)}_to_{int(hi)}_mean"] = _safe_stat(frac)
        feats[f"band_{int(lo)}_to_{int(hi)}_std"] = _safe_std(frac)

    # High-frequency energy ratio (> 8 kHz, if Nyquist allows).
    hf_mask = freqs > 8000.0
    if hf_mask.any():
        hf_frac = power[hf_mask, :].sum(axis=0) / total
        feats["hf_energy_ratio_mean"] = _safe_stat(hf_frac)
        feats["hf_energy_ratio_std"] = _safe_std(hf_frac)

    # ------------------------------------------------------------------ temporal
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=HOP)[0]
    feats["zcr_mean"] = _safe_stat(zcr)
    feats["zcr_std"] = _safe_std(zcr)

    rms = librosa.feature.rms(y=audio, hop_length=HOP)[0]
    rms = rms[rms > 0]
    feats["rms_mean"] = _safe_stat(rms)
    feats["rms_std"] = _safe_std(rms)
    if rms.size > 2:
        centered = (rms - rms.mean()) / (rms.std() + 1e-12)
        feats["rms_kurtosis"] = float(np.mean(centered ** 4) - 3.0)
        feats["rms_skew"] = float(np.mean(centered ** 3))
    else:
        feats["rms_kurtosis"] = None
        feats["rms_skew"] = None

    # Envelope modulation spectrum flatness (low => regular "pumping").
    env = librosa.feature.rms(y=audio, hop_length=HOP)[0]
    env = env - env.mean()
    if env.size > 8:
        env_spec = np.abs(np.fft.rfft(env)) ** 2
        env_spec = env_spec[1:]  # drop DC
        gmean = np.exp(np.mean(np.log(env_spec + 1e-12)))
        amean = np.mean(env_spec) + 1e-12
        feats["mod_flatness"] = float(gmean / amean)
    else:
        feats["mod_flatness"] = None

    # Onset regularity: coefficient of variation of inter-onset intervals.
    try:
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=HOP, n_fft=N_FFT)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, hop_length=HOP, backtrack=True
        )
        onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=HOP)
        if onset_times.size >= 4:
            ioi = np.diff(onset_times)
            ioi = ioi[ioi > 1e-3]
            if ioi.size >= 3 and np.mean(ioi) > 0:
                feats["onset_regularity"] = float(np.std(ioi) / np.mean(ioi))
            else:
                feats["onset_regularity"] = None
        else:
            feats["onset_regularity"] = None
    except Exception:
        feats["onset_regularity"] = None

    # Beat stability: coefficient of variation of beat intervals.
    try:
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr, hop_length=HOP)
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=HOP)
        if beat_times.size >= 4:
            intervals = np.diff(beat_times)
            intervals = intervals[intervals > 1e-3]
            if intervals.size >= 3 and np.mean(intervals) > 0:
                feats["beat_cv"] = float(np.std(intervals) / np.mean(intervals))
            else:
                feats["beat_cv"] = None
        else:
            feats["beat_cv"] = None
        feats["tempo_bpm"] = float(np.atleast_1d(tempo)[0]) if np.isfinite(np.atleast_1d(tempo)[0]) else None
    except Exception:
        feats["beat_cv"] = None
        feats["tempo_bpm"] = None

    # ------------------------------------------------------------- statistical
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=N_FFT, hop_length=HOP)
        for c in range(mfcc.shape[0]):
            feats[f"mfcc_mean_{c + 1}"] = _safe_stat(mfcc[c])
            feats[f"mfcc_std_{c + 1}"] = _safe_std(mfcc[c])
    except Exception:
        for c in range(13):
            feats[f"mfcc_mean_{c + 1}"] = None
            feats[f"mfcc_std_{c + 1}"] = None

    return feats
