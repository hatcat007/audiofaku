"""
Synthetic corpus generation (in-scope per SOW §3: "synthetic tone-based
test signals" and open-licensed material only).

Generates pairs of short WAV clips: a *sterile* version (perfect
quantization, constant velocity, static spectra — the kind of statistics
machine-made music exhibits) and a *humanized* version (onset jitter,
velocity variation, vibrato, noise floor, breathing envelope).  The pairs
let the harness sanity-check that its surrogate instruments actually
separate the two classes, before any sanitizer is applied.

All audio is synthesized locally with numpy; nothing is downloaded.
"""

from __future__ import annotations

import json
import numpy as np
import soundfile as sf
from pathlib import Path

SR_DEFAULT = 44100
MIDI_A4 = 69


def midi_to_freq(m: int) -> float:
    return float(440.0 * 2 ** ((m - MIDI_A4) / 12.0))


def _env(n: int, sr: int, attack: float, release: float) -> np.ndarray:
    """Simple attack/release amplitude envelope."""
    a = max(1, int(attack * sr))
    r = max(1, int(release * sr))
    env = np.ones(n)
    if n <= a + r:
        return np.linspace(0.0, 1.0, n)
    env[:a] = np.linspace(0.0, 1.0, a)
    env[-r:] = np.linspace(1.0, 0.0, r)
    return env


def _note(
    freq: float,
    dur: float,
    sr: int,
    velocity: float = 1.0,
    vibrato_depth: float = 0.0,
    vibrato_hz: float = 5.0,
    noise_ratio: float = 0.10,
    partials: tuple = (
        (1.0, 1.0),
        (2.0, 0.42),
        (3.0, 0.22),
        (4.0, 0.10),
        (5.0, 0.05),
    ),
    attack: float = 0.01,
    release: float = 0.08,
    start_jitter: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    A single synthetic note.

    Uses mildly inharmonic partial multipliers (detuned per-note) plus a
    low-level breath-noise component, so the corpus timbre behaves like
    real recorded instruments under timing perturbations (pure tones are
    pathologically sensitive to sub-ms shifts and make quality metrics
    meaningless).
    """
    rng = rng or np.random.default_rng(0)
    n = int(dur * sr)
    t = np.linspace(0.0, dur, n, endpoint=False)
    env = _env(n, sr, attack, release)

    sig = np.zeros(n)
    for mult, amp in partials:
        if freq * mult >= sr / 2.0 - 100:
            continue
        # per-partial random detune (stable within the note)
        detune = 1.0 + rng.normal(0.0, 0.0006)
        phase = 2.0 * np.pi * freq * mult * detune * t
        if vibrato_depth > 0:
            phase = phase + (2.0 * np.pi * freq * vibrato_depth / freq) * np.sin(
                2.0 * np.pi * vibrato_hz * t
            )
        sig += amp * np.sin(phase)

    # breath/bow noise shaped by the same envelope (high-passed white)
    if noise_ratio > 0:
        white = rng.normal(0.0, 1.0, n)
        breath = np.diff(white, prepend=0.0)  # crude high-pass
        sig += noise_ratio * breath * env

    sig *= env
    sig *= velocity
    if start_jitter > 0 and rng is not None:
        j = int(rng.uniform(-start_jitter, start_jitter) * sr)
        if j > 0:
            sig = np.concatenate([np.zeros(j), sig])[:n]
        elif j < 0:
            sig = np.concatenate([sig[-j:], np.zeros(-j)])[:n]
    return sig


def _brown_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    w = rng.normal(0.0, 1.0, n)
    b = np.cumsum(w)
    b /= (np.max(np.abs(b)) + 1e-12)
    return b


def _add_floor(sig: np.ndarray, level_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add a low-level noise floor and normalize to peak 0.9."""
    noise = _brown_noise(sig.size, rng)
    sig = sig + noise * (10 ** (level_db / 20.0))
    sig = sig / (np.max(np.abs(sig)) + 1e-12) * 0.9
    return sig


# --------------------------------------------------------------------------
# Sterile / humanized renderers.  Each returns mono float64 audio.
# --------------------------------------------------------------------------

def _render_chords(duration: float, sr: int, humanized: bool, rng: np.random.Generator) -> np.ndarray:
    progression = [
        [60, 64, 67],  # C
        [57, 60, 64],  # Am
        [53, 57, 60],  # F
        [55, 59, 62],  # G
    ]
    chord_dur = duration / len(progression)
    out = np.zeros(int(duration * sr))
    jitter = 0.018 if humanized else 0.0
    vel_jit = 0.20 if humanized else 0.0
    vib = 0.0035 if humanized else 0.0
    noise_ratio = 0.20 if humanized else 0.10
    for i, chord in enumerate(progression):
        start = i * chord_dur
        for m in chord:
            vel = 1.0 + (rng.uniform(-vel_jit, vel_jit) if humanized else 0.0)
            note = _note(
                midi_to_freq(m),
                chord_dur * 1.05,
                sr,
                velocity=vel,
                vibrato_depth=vib,
                noise_ratio=noise_ratio,
                start_jitter=jitter,
                rng=rng,
            )
            s = int(start * sr)
            e = min(len(out), s + len(note))
            out[s:e] += note[: e - s]
    # Breathing envelope (humanized only).
    if humanized:
        t = np.linspace(0.0, duration, len(out), endpoint=False)
        out *= 1.0 + 0.05 * np.sin(2 * np.pi * t / 7.3 + rng.uniform(0, 2 * np.pi))
    return out


def _render_arpeggio(duration: float, sr: int, humanized: bool, rng: np.random.Generator) -> np.ndarray:
    pattern = [60, 64, 67, 72, 67, 64]
    step = 60.0 / 120.0 / 2.0  # 8th notes @ 120 bpm
    out = np.zeros(int(duration * sr))
    jitter = 0.018 if humanized else 0.0
    vel_jit = 0.22 if humanized else 0.0
    vib = 0.004 if humanized else 0.0
    noise_ratio = 0.22 if humanized else 0.10
    t = 0.0
    i = 0
    while t < duration:
        m = pattern[i % len(pattern)]
        vel = 1.0 + (rng.uniform(-vel_jit, vel_jit) if humanized else 0.0)
        note = _note(
            midi_to_freq(m),
            step * 1.8,
            sr,
            velocity=vel,
            vibrato_depth=vib,
            noise_ratio=noise_ratio,
            start_jitter=jitter,
            rng=rng,
        )
        s = int(t * sr)
        e = min(len(out), s + len(note))
        out[s:e] += note[: e - s]
        t += step
        i += 1
    return out


def _render_percussion(duration: float, sr: int, humanized: bool, rng: np.random.Generator) -> np.ndarray:
    step = 60.0 / 120.0 / 4.0  # 16th notes @ 120 bpm
    n_steps = int(duration / step)
    out = np.zeros(int(duration * sr))

    def kick(d: float) -> np.ndarray:
        n = int(d * sr)
        t = np.linspace(0.0, d, n)
        freq = 55.0 * np.exp(-t * 8.0) + 30.0
        return np.sin(2 * np.pi * np.cumsum(freq) / sr) * np.exp(-t * 22.0)

    def snare(d: float) -> np.ndarray:
        n = int(d * sr)
        t = np.linspace(0.0, d, n)
        noise = rng.normal(0.0, 1.0, n) * np.exp(-t * 16.0)
        tone = np.sin(2 * np.pi * 190.0 * t) * np.exp(-t * 24.0)
        return 0.7 * noise + 0.3 * tone

    def hat(d: float) -> np.ndarray:
        n = int(d * sr)
        t = np.linspace(0.0, d, n)
        noise = rng.normal(0.0, 1.0, n)
        hp = np.diff(noise, prepend=0.0)  # crude high-pass
        return hp * np.exp(-t * 70.0)

    jitter = 0.012 if humanized else 0.0
    vel_jit = 0.22 if humanized else 0.0
    for i in range(n_steps):
        t_start = i * step + (rng.uniform(-jitter, jitter) if humanized else 0.0)
        s = max(0, int(t_start * sr))  # clamp: jitter can make step 0 start before 0
        if s >= len(out):
            break
        vel = 1.0 + (rng.uniform(-vel_jit, vel_jit) if humanized else 0.0)
        if i % 4 == 0:
            hit = kick(0.18) * vel
        elif i % 4 == 2:
            hit = snare(0.22) * vel
        else:
            hit = hat(0.06) * (vel * (0.7 if i % 2 == 0 else 0.9))
        e = min(len(out), s + len(hit))
        out[s:e] += hit[: e - s]
    return out


def _render_ambient(duration: float, sr: int, humanized: bool, rng: np.random.Generator) -> np.ndarray:
    freqs = [110.0, 165.0, 220.0, 330.0]
    n = int(duration * sr)
    t = np.linspace(0.0, duration, n, endpoint=False)
    out = np.zeros(n)
    for i, f in enumerate(freqs):
        lfo_rate = (0.08 + 0.02 * i) + (rng.uniform(-0.01, 0.01) if humanized else 0.0)
        lfo_phase = rng.uniform(0, 2 * np.pi)
        depth = 0.25 if humanized else 0.2
        amp = 0.22 * (1.0 + depth * np.sin(2 * np.pi * lfo_rate * t + lfo_phase))
        detune = rng.uniform(-0.4, 0.4) if humanized else 0.0
        phase = 2 * np.pi * (f + detune) * t
        if humanized:
            phase = phase + 0.02 * np.sin(2 * np.pi * 0.05 * t + lfo_phase)
        out += amp * np.sin(phase)
    if humanized:
        # Slow random walk on the master level.
        walk = np.cumsum(rng.normal(0, 0.0002, n))
        out *= 1.0 + np.clip(walk, -0.15, 0.15)
    return out


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

_CORPUS_SPECS = [
    ("chords_sterile.wav", "sterile", "chord progression, perfect quantization"),
    ("chords_humanized.wav", "humanized", "chord progression, jitter/velocity/vibrato"),
    ("arpeggio_sterile.wav", "sterile", "8th-note arpeggio @120bpm, grid-quantized"),
    ("arpeggio_humanized.wav", "humanized", "8th-note arpeggio, humanized timing"),
    ("percussion_sterile.wav", "sterile", "16th-note drum pattern, grid-quantized"),
    ("percussion_humanized.wav", "humanized", "16th-note drum pattern, humanized"),
    ("ambient_sterile.wav", "sterile", "LFO-modulated sine pads, static LFOs"),
    ("ambient_humanized.wav", "humanized", "LFO-modulated sine pads, drifting LFOs"),
]


def generate_corpus(out_dir: Path | str, duration: float = 10.0, sr: int = SR_DEFAULT, seed: int = 42) -> dict:
    """
    Generate the synthetic control corpus into ``out_dir``.

    Returns the manifest dict (also written to ``out_dir/manifest.json``).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    files = []
    for fname, cls, desc in _CORPUS_SPECS:
        humanized = cls == "humanized"
        if fname.startswith("chords"):
            audio = _render_chords(duration, sr, humanized, rng)
        elif fname.startswith("arpeggio"):
            audio = _render_arpeggio(duration, sr, humanized, rng)
        elif fname.startswith("percussion"):
            audio = _render_percussion(duration, sr, humanized, rng)
        else:
            audio = _render_ambient(duration, sr, humanized, rng)
        audio = _add_floor(audio, -46.0 if humanized else -60.0, rng)
        path = out_dir / fname
        sf.write(str(path), audio, sr, subtype="PCM_16")
        files.append({"path": fname, "class": cls, "desc": desc})

    manifest = {
        "generator": "mmm.research.corpus",
        "sr": sr,
        "duration_seconds": duration,
        "seed": seed,
        "license_note": "synthetically generated in-process; no copyrighted material",
        "files": files,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def discover_inputs(paths: list[str] | None, corpus_dir: Path | str) -> tuple[list[Path], dict]:
    """
    Resolve the input file list.

    If ``paths`` is provided, expand files/directories to audio files
    (wav/mp3/flac/ogg).  Otherwise use the generated corpus in
    ``corpus_dir``.  Returns (files, manifest_or_empty).
    """
    corpus_dir = Path(corpus_dir)
    manifest: dict = {}
    if paths:
        files: list[Path] = []
        exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        for p in paths:
            p = Path(p)
            if p.is_dir():
                files.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in exts))
            elif p.is_file() and p.suffix.lower() in exts:
                files.append(p)
        return sorted(files), manifest

    if (corpus_dir / "manifest.json").exists():
        with open(corpus_dir / "manifest.json", encoding="utf-8") as fh:
            manifest = json.load(fh)
        return sorted(corpus_dir / f["path"] for f in manifest["files"]), manifest

    manifest = generate_corpus(corpus_dir)
    return sorted(corpus_dir / f["path"] for f in manifest["files"]), manifest
