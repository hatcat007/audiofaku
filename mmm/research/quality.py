"""
Objective quality metrics between an original and a processed clip.

Implements SNR, log-spectral distance, spectral angle, relative L1/L2
distance, RMS ratio, and — when the optional libraries are installed —
STOI (speech-oriented, informational only).  PEAQ/PESQ are not bundled
(dependencies unavailable in the offline environment); the SOW's
authoritative quality instruments are the sponsors' evaluation libraries.
"""

from __future__ import annotations

import numpy as np
import librosa

N_FFT = 2048
HOP = 512


def _align(orig: np.ndarray, proc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(orig.size, proc.size)
    return orig[:n], proc[:n]


def _stft_db(x: np.ndarray) -> np.ndarray:
    S = np.abs(librosa.stft(x, n_fft=N_FFT, hop_length=HOP))
    return 20.0 * np.log10(S + 1e-10)


def quality_metrics(orig: np.ndarray, proc: np.ndarray, sr: int) -> dict:
    """
    Compute objective quality metrics.

    All inputs mono float64.  Returns dict with float values, or None
    entries for metrics that could not be computed (missing optional
    dependency, incompatible sample rate, etc.).
    """
    orig = np.asarray(orig, dtype=np.float64)
    proc = np.asarray(proc, dtype=np.float64)
    if orig.ndim > 1:
        orig = orig.mean(axis=1)
    if proc.ndim > 1:
        proc = proc.mean(axis=1)

    orig, proc = _align(orig, proc)
    out: dict = {}

    if orig.size == 0 or proc.size == 0:
        return {"error": "empty signal"}

    # SNR: signal-to-(processing-)noise ratio.
    noise = orig - proc
    sig_pow = float(np.mean(orig ** 2))
    noi_pow = float(np.mean(noise ** 2))
    out["snr_db"] = float(np.clip(10.0 * np.log10(sig_pow / (noi_pow + 1e-12)), -40.0, 120.0))

    # Aligned SNR: find the best whole-signal lag (cross-correlation on a
    # 1 s probe window, ±100 ms search) and compute SNR after alignment.
    # Timing-oriented sanitizer stages (tempo drift, transient shifts) can
    # lower raw SNR substantially while preserving perceived quality;
    # aligned SNR separates "content preserved but time-shifted" from
    # "content destroyed".
    try:
        probe = min(len(orig), sr)
        seg_o = orig[:probe] - orig[:probe].mean()
        seg_p = proc[:probe] - proc[:probe].mean()
        max_lag = int(0.1 * sr)
        corr = np.correlate(seg_p, seg_o, mode="full")
        peak = int(np.argmax(corr))
        lag = peak - probe + 1
        lag = int(np.clip(lag, -max_lag, max_lag))
        if lag >= 0:
            a_o, a_p = orig[lag:], proc[: len(orig) - lag]
        else:
            a_o, a_p = orig[: lag], proc[-lag:]
        n_align = min(len(a_o), len(a_p))
        if n_align > sr:
            aligned_noise = a_o[:n_align] - a_p[:n_align]
            out["aligned_snr_db"] = float(
                np.clip(
                    10.0
                    * np.log10(
                        np.mean(a_o[:n_align] ** 2) / (np.mean(aligned_noise ** 2) + 1e-12)
                    ),
                    -40.0,
                    120.0,
                )
            )
            out["alignment_lag_ms"] = round(1000.0 * lag / sr, 2)
        else:
            out["aligned_snr_db"] = None
            out["alignment_lag_ms"] = None
    except Exception:
        out["aligned_snr_db"] = None
        out["alignment_lag_ms"] = None

    # Log-spectral distance between *mean* dB spectra (in dB), masked to
    # bins with meaningful energy in the original (>= -60 dB relative to
    # its peak bin).  Averaging over frames first makes this robust to
    # timing/phase alterations; the mask keeps the silence-region noise
    # floor from dominating the distance.
    m_db1 = _stft_db(orig).mean(axis=1)
    m_db2 = _stft_db(proc).mean(axis=1)
    peak_db = np.max(m_db1) if m_db1.size else -200.0
    mask = m_db1 >= (peak_db - 60.0)
    if mask.any():
        out["mean_lsd_db"] = float(np.sqrt(np.mean((m_db1[mask] - m_db2[mask]) ** 2)))
    else:
        out["mean_lsd_db"] = None

    # Per-frame log-spectral distance (timing-sensitive; a time warp shows
    # up as a large value here even when the spectrum is preserved).
    S1, S2 = _stft_db(orig), _stft_db(proc)
    out["frame_lsd_db"] = float(np.mean(np.sqrt(np.mean((S1 - S2) ** 2, axis=0))))

    # Spectral angle between mean *power* spectra (1 = identical direction).
    # Power-domain averaging is dominated by the tonal content rather than
    # the noise floor, so it stays meaningful for sparse-spectrum material.
    P1 = (10.0 ** (S1 / 10.0)).mean(axis=1)
    P2 = (10.0 ** (S2 / 10.0)).mean(axis=1)
    denom = (np.linalg.norm(P1) * np.linalg.norm(P2)) + 1e-12
    out["spectral_angle_cos"] = float(np.clip(np.dot(P1, P2) / denom, -1.0, 1.0))

    # MFCC timbre similarity (cosine between mean MFCC vectors).
    try:
        mfcc1 = librosa.feature.mfcc(y=orig, sr=sr, n_mfcc=13, n_fft=N_FFT, hop_length=HOP)
        mfcc2 = librosa.feature.mfcc(y=proc, sr=sr, n_mfcc=13, n_fft=N_FFT, hop_length=HOP)
        v1 = mfcc1.mean(axis=1)
        v2 = mfcc2.mean(axis=1)
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-12
        out["mfcc_cos"] = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
    except Exception:
        out["mfcc_cos"] = None

    # Relative distances.
    out["l1_rel"] = float(np.mean(np.abs(noise)) / (np.mean(np.abs(orig)) + 1e-12))
    out["l2_rel"] = float(np.linalg.norm(noise) / (np.linalg.norm(orig) + 1e-12))
    out["rms_ratio"] = float(np.sqrt(np.mean(proc ** 2)) / (np.sqrt(np.mean(orig ** 2)) + 1e-12))

    # Locally-aligned SNR: split both signals into 100 ms frames, find the
    # best ±50 ms lag per frame (cross-correlation), align, and compute SNR
    # per frame; report the median.  This compensates for the sanitizer's
    # intentional micro-timing changes (tempo drift accumulates offsets up
    # to ~10 ms on short clips; transient shifts are per-event) and is the
    # most meaningful time-domain fidelity number for this tool.
    try:
        win = int(0.1 * sr)
        hop = win // 2
        maxlag = int(0.05 * sr)
        snrs = []
        limit = min(len(orig), len(proc)) - win
        for start in range(0, limit, hop):
            a = orig[start : start + win]
            b = proc[start : start + win]
            a = a - a.mean()
            b = b - b.mean()
            c = np.correlate(b, a, mode="full")
            lag = int(np.clip(int(np.argmax(c)) - win + 1, -maxlag, maxlag))
            if lag >= 0:
                a2, b2 = a[lag:], b[: win - lag]
            else:
                a2, b2 = a[:lag], b[-lag:]
            nn = min(len(a2), len(b2))
            if nn < win // 2:
                continue
            noise = a2[:nn] - b2[:nn]
            snr = 10.0 * np.log10(np.mean(a2[:nn] ** 2) / (np.mean(noise ** 2) + 1e-12))
            snrs.append(snr)
        out["local_aligned_snr_db"] = float(np.median(snrs)) if snrs else None
    except Exception:
        out["local_aligned_snr_db"] = None

    # Warp-compensated SNR: estimate the time-warp curve between proc and
    # orig (per-frame best lag, smoothed), resample proc to undo it, and
    # compute SNR on the compensated signal.  The sanitizer's *documented*
    # behavior includes intentional micro-timing changes (tempo drift,
    # resample warps); this metric separates "timing changed" from
    # "content changed".  Per-event transient shifts that are not smooth
    # in time are not fully undone — which is the honest behavior.
    try:
        fwin = int(0.05 * sr)
        fhop = fwin // 2
        search = int(0.15 * sr)
        lags = []
        centers = []
        limit = min(len(orig), len(proc)) - fwin
        for start in range(0, limit, fhop):
            pf = proc[start : start + fwin]
            ctx_start = max(0, start - search)
            oc = orig[ctx_start : start + fwin + search]
            pf = pf - pf.mean()
            oc = oc - oc.mean()
            if pf.size == 0 or oc.size < pf.size:
                continue
            # c[k] aligns proc[start] with orig[ctx_start + k]; the lag is
            # the orig index minus the proc index.
            c = np.correlate(pf, oc, mode="valid")
            k = int(np.argmax(c))
            lag = ctx_start + k - start
            lags.append(lag)
            centers.append(start + fwin // 2)
        if len(lags) >= 8:
            lags = np.asarray(lags, dtype=np.float64)
            centers = np.asarray(centers, dtype=np.float64)
            # reject outlier lags (correlation failures)
            med = np.median(lags)
            mad = np.median(np.abs(lags - med)) + 1e-9
            good = np.abs(lags - med) < 10.0 * mad
            if good.sum() >= 8:
                lags, centers = lags[good], centers[good]
                lag_curve = np.interp(
                    np.arange(len(orig)), centers, lags, left=lags[0], right=lags[-1]
                )
                warp_map = np.clip(np.arange(len(orig)) + lag_curve, 0, len(orig) - 1)
                compensated = np.interp(
                    warp_map, np.arange(len(proc)).astype(np.float64), proc
                )
                noise = orig - compensated
                out["warp_snr_db"] = float(
                    np.clip(
                        10.0
                        * np.log10(
                            np.mean(orig ** 2) / (np.mean(noise ** 2) + 1e-12)
                        ),
                        -40.0,
                        120.0,
                    )
                )
                out["warp_max_lag_ms"] = round(
                    1000.0 * float(np.max(np.abs(lag_curve))) / sr, 2
                )
            else:
                out["warp_snr_db"] = None
                out["warp_max_lag_ms"] = None
        else:
            out["warp_snr_db"] = None
            out["warp_max_lag_ms"] = None
    except Exception:
        out["warp_snr_db"] = None
        out["warp_max_lag_ms"] = None

    # STOI (speech-oriented; informational only).  Requires pystoi.
    try:
        from pystoi import stoi  # type: ignore

        out["stoi"] = float(np.clip(stoi(orig, proc, sr, extended=False), 0.0, 1.0))
    except Exception:
        out["stoi"] = None

    # PESQ (telephony-oriented, 8/16 kHz only).  Requires the `pesq` package.
    try:
        from pesq import pesq as _pesq  # type: ignore

        if sr in (8000, 16000):
            out["pesq"] = float(np.clip(_pesq(sr, orig, proc, "wb"), -0.5, 4.5))
        else:
            out["pesq"] = None
    except Exception:
        out["pesq"] = None

    return out
