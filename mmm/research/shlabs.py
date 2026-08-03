"""
SHLabs-family feature extraction (surrogate for the SubmitHub checker's
documented feature set).

Public source documents (SubmitHub engineering blog, v3/v4) describe the
checker's features:

1. Basic Spectral: spectral flatness, spectral rolloff, RMS energy,
   zero-crossing rate, MFCC statistics.
2. Harmonic: phase coherence, frequency-band ratios, harmonic
   consistency, harmonic stability, pitch transition rate, harmonic
   complexity, vocals/music ratio.
3. Long-range pattern analysis: section correlation/variation over
   60/120/180-second windows (per the SONICS paper, ICLR 2025, long-range
   temporal dependencies are the key signal).

v4 additionally describes the spectral analysis as "turning each song
into a picture" (spectrogram image analysis) — we approximate the image-
texture statistics with DCT-based spectral-texture features.

Every feature here is a documented surrogate.  Where a feature can only
be approximated cheaply (vocals/music ratio, pitch transition rate), the
approximation is labeled and described.
"""

from __future__ import annotations

import numpy as np
import librosa
from scipy.fft import dct as _dct

N_FFT = 2048
HOP = 512


def _safe_mean(x: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else None


def _safe_std(x: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.std()) if x.size else None


def _gini(x: np.ndarray) -> float:
    """Gini coefficient of |x| (0 = uniform, 1 = concentrated)."""
    x = np.abs(np.asarray(x, dtype=np.float64)).ravel()
    if x.size == 0 or x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / (cum[-1] + 1e-12)) / n)


def extract_sh_features(audio: np.ndarray, sr: int) -> dict:
    """
    Extract the SH-family feature set from mono audio.

    Returns dict[str, float | None]; None = not computable (too short
    for long-range windows, no peaks, etc).
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.size < N_FFT:
        raise ValueError("audio too short")

    audio = audio / (np.max(np.abs(audio)) + 1e-12)
    n = audio.size
    duration = n / sr
    out: dict[str, float | None] = {}

    S = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    power = S ** 2
    total = power.sum(axis=0) + 1e-12

    # ------------------------------------------------------------ 1. spectral
    flat = librosa.feature.spectral_flatness(S=S)[0]
    roll85 = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0]
    roll95 = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.95)[0]
    rms = librosa.feature.rms(y=audio, hop_length=HOP)[0]
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=HOP)[0]
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=N_FFT, hop_length=HOP)

    out["sh_flatness_mean"] = _safe_mean(flat)
    out["sh_flatness_std"] = _safe_std(flat)
    out["sh_rolloff85_mean"] = _safe_mean(roll85)
    out["sh_rolloff95_mean"] = _safe_mean(roll95)
    out["sh_rolloff85_std"] = _safe_std(roll85)
    out["sh_rms_mean"] = _safe_mean(rms)
    out["sh_rms_std"] = _safe_std(rms)
    out["sh_rms_kurt"] = float(
        np.mean(((rms - rms.mean()) / (rms.std() + 1e-12)) ** 4) - 3.0
    )
    out["sh_zcr_mean"] = _safe_mean(zcr)
    out["sh_zcr_std"] = _safe_std(zcr)
    for c in range(13):
        out[f"sh_mfcc_mean_{c}"] = _safe_mean(mfcc[c])
        out[f"sh_mfcc_std_{c}"] = _safe_std(mfcc[c])

    # ------------------------------------------------------------ 2. harmonic
    # Phase coherence: per-bin mean phasor magnitude over time (1 = phase
    # perfectly consistent across time).  Averaged over 50-4000 Hz.
    phase = np.angle(S)
    band = (freqs >= 50) & (freqs <= 4000)
    if band.any():
        mean_phasor = np.abs(np.mean(np.exp(1j * phase[band, :]), axis=1))
        out["sh_phase_coherence"] = _safe_mean(mean_phasor)
    else:
        out["sh_phase_coherence"] = None

    # Frequency band ratios: energy fractions in standard zones.
    edges = [0, 250, 500, 1000, 2000, 4000, 8000, 12000, sr / 2 + 1]
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        mask = (freqs >= lo) & (freqs < hi)
        if not mask.any():
            continue
        frac = power[mask, :].sum(axis=0) / total
        out[f"sh_band_{lo}_{hi}_mean"] = _safe_mean(frac)
        out[f"sh_band_{lo}_{hi}_std"] = _safe_std(frac)

    # Chroma-based harmonic features.
    hop_c = 4096
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=N_FFT, hop_length=hop_c)
    n_frames_c = chroma.shape[1]

    # Harmonic consistency: mean correlation of consecutive 10 s chroma blocks.
    block = max(1, int(10.0 * sr / hop_c))
    blocks = []
    for b in range(0, n_frames_c - block + 1, block):
        blocks.append(chroma[:, b : b + block].mean(axis=1))
    if len(blocks) >= 3:
        corrs = [
            float(np.corrcoef(blocks[i], blocks[i + 1])[0, 1])
            for i in range(len(blocks) - 1)
        ]
        out["sh_harmonic_consistency"] = _safe_mean(corrs)
        out["sh_harmonic_stability"] = _safe_std(np.asarray(blocks).mean(axis=1))
    else:
        out["sh_harmonic_consistency"] = None
        out["sh_harmonic_stability"] = None

    # Pitch transition rate: chroma argmax transitions per second.
    if n_frames_c > 2:
        argmax = np.argmax(chroma, axis=0)
        transitions = int(np.sum(np.diff(argmax) != 0))
        out["sh_pitch_transition_rate"] = float(transitions / max(duration, 1e-9))
    else:
        out["sh_pitch_transition_rate"] = None

    # Harmonic complexity: number of distinct spectral peaks > -40 dB rel peak.
    mean_spec_db = 20 * np.log10(S.mean(axis=1) + 1e-10)
    band_m = (freqs > 100) & (freqs < 5000)
    spec_b = mean_spec_db[band_m]
    if spec_b.size:
        thresh = np.max(spec_b) - 40.0
        peaks = 0
        for i in range(1, spec_b.size - 1):
            if spec_b[i] > spec_b[i - 1] and spec_b[i] >= spec_b[i + 1] and spec_b[i] > thresh:
                peaks += 1
        out["sh_harmonic_complexity"] = float(peaks)
    else:
        out["sh_harmonic_complexity"] = None

    # Vocals/music ratio (CRUDE approximation): energy fraction in the
    # vocal-dominant band (200-4000 Hz) modulated at 3-8 Hz (syllable/
    # vibrato rate).  Documented as an approximation of the real feature.
    vmask = (freqs >= 200) & (freqs <= 4000)
    if vmask.any():
        # envelope of vocal band energy over time
        vb_energy = power[vmask, :].sum(axis=0) / (total + 1e-12)
        # modulation energy in 3-8 Hz
        env = vb_energy - vb_energy.mean()
        if env.size > 16:
            spec_env = np.abs(np.fft.rfft(env)) ** 2
            freqs_env = np.fft.rfftfreq(env.size, d=hop_c / sr)
            mod_band = (freqs_env >= 3) & (freqs_env <= 8)
            mod_energy = spec_env[mod_band].sum() if mod_band.any() else 0.0
            mod_total = spec_env.sum() + 1e-12
            out["sh_vocals_ratio_crude"] = float(
                np.clip(vb_energy.mean() * (1.0 + 10.0 * mod_energy / mod_total), 0.0, 1.0)
            )
        else:
            out["sh_vocals_ratio_crude"] = float(np.clip(vb_energy.mean(), 0.0, 1.0))
    else:
        out["sh_vocals_ratio_crude"] = None

    # ------------------------------------------------ 3. spectral texture (v4
    # "picture" approximation): DCT of the downsampled log-mel spectrogram.
    try:
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64, n_fft=N_FFT, hop_length=HOP)
        logmel = librosa.power_to_db(np.maximum(mel, 1e-10))
        # downsample time axis to <= 256 frames
        step = max(1, logmel.shape[1] // 256)
        logmel = logmel[:, ::step][:, :256]
        # 2D DCT via separable 1D DCTs (type-II)
        dct_rows = np.apply_along_axis(
            lambda v: np.abs(_dct(v, type=2, norm="ortho")), 0, logmel
        )
        dct2d = np.apply_along_axis(
            lambda v: np.abs(_dct(v, type=2, norm="ortho")), 1, dct_rows
        )
        dct2d = dct2d[1:, 1:]  # drop DC row/col
        out["sh_texture_gini"] = _gini(dct2d)
        # low-frequency DCT energy fraction (smoothness)
        low = dct2d[:16, :16].sum()
        out["sh_texture_smoothness"] = float(low / (dct2d.sum() + 1e-12))
        # block-wise variance of log-mel (local texture roughness)
        blk = logmel[:16, :16]
        out["sh_texture_local_var"] = _safe_std(blk)
    except Exception:
        out["sh_texture_gini"] = None
        out["sh_texture_smoothness"] = None
        out["sh_texture_local_var"] = None

    # Tempo stability across 30 s chunks (the checker's temporal side is
    # documented as "tempo, phase and timing alignment"; the sanitizer's
    # long-range tempo drift directly targets this).
    try:
        chunk_s = 30.0
        chunk = int(chunk_s * sr)
        tempi = []
        for st in range(0, n - chunk + 1, chunk):
            seg = audio[st : st + chunk]
            t_bpm = librosa.beat.beat_track(y=seg, sr=sr)[0]
            t_bpm = float(np.atleast_1d(t_bpm)[0])
            if np.isfinite(t_bpm) and 40.0 < t_bpm < 220.0:
                tempi.append(t_bpm)
        if len(tempi) >= 3:
            out["sh_tempo_chunk_std"] = _safe_std(tempi)
            out["sh_tempo_chunk_mean"] = _safe_mean(tempi)
        else:
            out["sh_tempo_chunk_std"] = None
            out["sh_tempo_chunk_mean"] = None
    except Exception:
        out["sh_tempo_chunk_std"] = None
        out["sh_tempo_chunk_mean"] = None

    # ------------------------------------------------ 4. long-range section
    # correlation/variation over 60/120/180 s windows (SONICS insight).
    hop_lr = 2048
    mel_lr = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=32, n_fft=N_FFT, hop_length=hop_lr)
    logmel_lr = librosa.power_to_db(np.maximum(mel_lr, 1e-10))

    def section_stats(window_s: int) -> None:
        win = int(window_s * sr / hop_lr)
        if win < 2 or n_frames_c < 3:
            out[f"sh_section_corr_{window_s}"] = None
            out[f"sh_section_std_{window_s}"] = None
            return
        sections = []
        for st in range(0, logmel_lr.shape[1] - win + 1, win):
            sections.append(logmel_lr[:, st : st + win].mean(axis=1))
        if len(sections) < 2:
            out[f"sh_section_corr_{window_s}"] = None
            out[f"sh_section_std_{window_s}"] = None
            return
        corrs = []
        for i in range(len(sections)):
            for j in range(i + 1, len(sections)):
                c = np.corrcoef(sections[i], sections[j])[0, 1]
                if np.isfinite(c):
                    corrs.append(c)
        out[f"sh_section_corr_{window_s}"] = _safe_mean(corrs) if corrs else None
        # how much the sections vary
        sec_rms = [np.sqrt(np.mean(s ** 2)) for s in sections]
        out[f"sh_section_std_{window_s}"] = _safe_std(sec_rms)

    for w in (60, 120, 180):
        if duration >= w:
            section_stats(w)
        else:
            out[f"sh_section_corr_{w}"] = None
            out[f"sh_section_std_{w}"] = None

    # first-half vs second-half correlation (whole-track structure)
    half = logmel_lr.shape[1] // 2
    if half >= 8:
        h1 = logmel_lr[:, :half].mean(axis=1)
        h2 = logmel_lr[:, half : 2 * half].mean(axis=1)
        out["sh_half_half_corr"] = float(np.corrcoef(h1, h2)[0, 1])
    else:
        out["sh_half_half_corr"] = None

    return out
