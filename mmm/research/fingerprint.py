"""
Acoustic fingerprinting surrogate (Shazam-style spectral-peak hashing).

Stand-in for AHA Music / ACRCloud-style identification fingerprinting,
implemented in pure numpy so the harness runs fully offline.  The hash
construction follows the classic landmark-hashing approach:

1. Compute a magnitude spectrogram.
2. Pick the strongest local-maximum bins per frame within a bounded
   frequency band (40 Hz - 6 kHz, matching typical music-fingerprint
   ranges).
3. For each anchor peak, combine it with the next ``fanout`` peaks in the
   following frames:  hash = (anchor_bin, target_bin, delta_time).

Match confidence between an original and a processed clip is the fraction
of the original's hashes that survive in the processed clip (coverage).
Coverage 1.0 means "identical fingerprint"; coverage ~0 means the
fingerprint was destroyed.

This is a *surrogate*: the sponsors' sandboxed AHA/ACRCloud API is the
authoritative instrument.  The surrogate exists to measure ΔMatch
directionally before that API is wired in.  When the API is available,
swap ``fingerprint_match`` for the sponsor's matcher — the pipeline
interface is unchanged.
"""

from __future__ import annotations

import numpy as np
import librosa

MIN_FREQ = 40.0
MAX_FREQ = 6000.0

# hash packing: anchor_bin (10 bits) | target_bin (10 bits) | dt (6 bits)
_BIN_BITS = 10
_DT_BITS = 6
_BIN_MASK = (1 << _BIN_BITS) - 1
_DT_MASK = (1 << _DT_BITS) - 1


def spectral_peak_hashes(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop: int = 256,
    peaks_per_frame: int = 4,
    fanout: int = 3,
    max_dt_frames: int = 16,
    max_hashes: int = 1_500_000,
    seed: int | None = None,
) -> frozenset:
    """
    Compute a set of landmark hashes for a mono signal.

    The hash includes a 3-bit quantization of the anchor peak's magnitude
    (relative to the frame maximum) to make the fingerprint more
    discriminative for tonal material, where the same few spectral peaks
    persist across frames and otherwise dominate the hash set.

    Memory guard: on long tracks the combinatoric hash set can grow to
    tens of millions of Python ints (multi-GB).  ``max_hashes`` caps the
    set size — once reached, no further hashes are added.  The cap
    preserves the fingerprint's *early-track* structure, which is what
    the coverage metric compares.

    Returns a frozenset of int hashes.  Empty/too-short input yields an
    empty set.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.size < n_fft:
        return frozenset()

    audio = audio / (np.max(np.abs(audio)) + 1e-12)

    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band_mask = (freqs >= MIN_FREQ) & (freqs <= min(MAX_FREQ, sr / 2.0))
    band_idx = np.where(band_mask)[0]
    if band_idx.size < 2:
        return frozenset()

    S = S[band_idx, :]
    n_frames = S.shape[1]

    # Local maxima (bin is >= its immediate neighbors within the band).
    padded = np.pad(S, ((1, 1), (0, 0)), mode="edge")
    is_peak = (S >= padded[:-2, :]) & (S > padded[2:, :])

    # Per-frame top-k peak selection.
    hashes = set()
    peak_lists = []  # per frame: list of (band_position, magnitude)
    for t in range(n_frames):
        peaks = np.where(is_peak[:, t])[0]
        if peaks.size == 0:
            peak_lists.append([])
            continue
        mags = S[peaks, t]
        k = min(peaks_per_frame, peaks.size)
        top = np.argpartition(mags, -k)[-k:]
        peak_lists.append([(int(peaks[i]), float(mags[i])) for i in top])

    # Combinatoric hashing: anchor peak at frame t with peaks in t+1..t+max_dt.
    for t in range(n_frames):
        anchors = peak_lists[t]
        if not anchors:
            continue
        horizon = min(n_frames, t + 1 + max_dt_frames)
        targets: list[tuple[int, float, int]] = []  # (band_position, mag, dt)
        for u in range(t + 1, horizon):
            for t_bin, t_mag in peak_lists[u]:
                targets.append((t_bin, t_mag, u - t))
        if not targets:
            continue
        # Keep the strongest targets to bound hash count.
        if len(targets) > fanout * 8:
            mags = np.array([m for _, m, _ in targets])
            keep = np.argpartition(mags, -fanout * 8)[-fanout * 8:]
            targets = [targets[i] for i in keep]
        for a_bin, a_mag in anchors:
            # 3-bit anchor magnitude quantization (relative to frame max).
            mag_q = int(np.clip(a_mag / (np.max(S[:, t]) + 1e-12) * 7.0, 0, 7))
            for t_bin, _, dt_raw in targets:
                dt = min(max(dt_raw, 1), _DT_MASK)
                h = (
                    (a_bin << (_BIN_BITS + _DT_BITS + 3))
                    | (t_bin << (_DT_BITS + 3))
                    | (dt << 3)
                    | mag_q
                )
                hashes.add(h)
        if len(hashes) >= max_hashes:
            break

    return frozenset(hashes)


def match_coverage(original_hashes: frozenset, processed_hashes: frozenset) -> float:
    """
    Fraction of the original fingerprint surviving in the processed clip.

    1.0 = full match (fingerprint intact), 0.0 = no hashes survived.
    """
    if not original_hashes:
        return 0.0
    if not processed_hashes:
        return 0.0
    return len(original_hashes & processed_hashes) / len(original_hashes)


def specificity_check(hashes_a: frozenset, hashes_b: frozenset) -> float:
    """
    Match coverage between two *unrelated* clips — a false-positive control.
    Should be near 0 for a well-behaved fingerprint; report it so the
    reader can judge the surrogate's selectivity.
    """
    return match_coverage(hashes_a, hashes_b)
