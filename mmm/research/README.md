# mmm.research — Offline Robustness Benchmark Harness

Part of engagement **SUB-AHA-2026-SEC-001** (Robustness Assessment of Audio
Identification and AI-Generated Content Detection) — *offline phase*.

This package implements the sandboxed measurement loop described in the
engagement's Scope of Work:

```
baseline measurement  →  MMM sanitization (real code paths)  →  re-measurement
```

## Scope boundary (read this first)

Everything here runs **fully offline** on **synthetic / open-licensed**
audio. It does **not** call SubmitHub's AI Song Checker, AHA Music, or any
ACRCloud production service — per the engagement's out-of-scope list
("Live production testing on SubmitHub or AHA Music public services",
"Use of copyrighted or personally identifiable audio").

The two instruments are **surrogates**, clearly labeled as such:

| SOW quantity | Surrogate in this harness | Authoritative instrument |
|---|---|---|
| ΔMatch (fingerprint confidence) | spectral-peak landmark hashing (`fingerprint.py`) | sponsors' sandboxed AHA/ACRCloud API |
| ΔAI score | interpretable feature-direction proxy (`proxy.py`) | sponsors' SubmitHub evaluation library |

Swap-in points: replace `mmm/research/fingerprint.py`'s
`spectral_peak_hashes`/`match_coverage` and `proxy.py`'s
`score_ai_likeness` with the sponsor-provided libraries; the pipeline and
report interfaces stay the same.

## Usage

```bash
# venv with repo dependencies (see repo README)
source mmm_env/bin/activate

# default: generate 8-clip synthetic corpus (30 s), run preserving + paranoid
python -m mmm.research

# all documented variants (preserving, paranoid, max, stealth_plus, fast, aggressive)
python -m mmm.research --variants all

# your own open-licensed audio (FMA, LibriSpeech, ...)
python -m mmm.research --inputs ./mytrack.wav ./mydir

# longer clips, custom output
python -m mmm.research --duration 20 --out bench_results/run2
```

Outputs under `bench_results/`:

- `corpus/` — generated synthetic control clips + `manifest.json`
- `audio/<variant>/` — sanitized outputs
- `logs/<file>__<variant>.log` — captured sanitizer console output
- `results.json` — full machine-readable results (features included)
- `report.md` — human-readable report with sanity checks, per-file tables,
  aggregate summary, limitations, and the AI Disclosure Annex (SOW §7)

## What the report contains

1. **Corpus** — files and their control classes (sterile / humanized).
2. **Instrument sanity checks** — does the proxy separate the synthetic
   control classes before processing? Does the fingerprint produce false
   matches between unrelated clips?
3. **Per-file results** — ΔAI proxy (overall/spectral/temporal),
   fingerprint coverage + ΔMatch, SNR, LSD, spectral angle, STOI.
4. **Aggregate summary** — means per variant.
5. **Interpretation & limitations** — surrogate caveats.
6. **AI Disclosure Annex** — SOW §7 log of AI tool usage.

## Calibrated proxy v2 (SH-family) and stage ablation

`proxy_v2.py` implements the calibrated proxy built on the feature family
the SubmitHub checker itself documents (v3 engineering blog + SONICS paper,
ICLR 2025):

- **Basic spectral**: flatness, rolloff, RMS, ZCR, MFCC statistics
- **Harmonic**: phase coherence, band ratios, harmonic
  consistency/stability, pitch transition rate, harmonic complexity,
  vocals/music ratio (crude approximation — labeled)
- **Long-range** (SONICS): section correlation/variation over 60/120/180 s
  windows, tempo stability over 30 s chunks
- **Spectral texture** (v4 "picture" approximation): DCT-based statistics
  of the log-mel spectrogram

The proxy is *calibrated* with a two-point affine fit against two real
SubmitHub checker results (06_fast: spectral 87 / temporal 92;
05_stealth_plus: spectral 84 / temporal 44).  The temporal slope is
regularized (clamped) because a raw two-point fit is implausibly steep —
the current temporal feature family only partially captures what the
checker's temporal classifier responds to.  Refit as more anchor points
become available: `calibrate()` reads a features dict and writes
`calibration.json`.

`ablation.py` applies each sanitizer stage *alone* to a track and measures
proxy deltas (raw + calibrated), fingerprint coverage, and quality
(warp-SNR, masked mean-LSD, spectral angle).  Usage:

```bash
python -m mmm.research.ablation --input track.wav --out bench_results/ablation
# --resume keeps already-measured entries; results in ablation.json
```

Stage outputs are written as WAVs under `ablation/audio/` for listening
and for spot-checking against the sponsors' sandboxed API.

## Reading the numbers honestly

- **ΔAI proxy** is a *directional* surrogate. A negative ΔAI proxy means
  the processed clip reads less machine-like to *this proxy* — it is not
  a SubmitHub score and is not evidence about the live checker.
- **Calibrated deltas** are only as good as the anchor points.  With two
  anchors the spectral calibration may even be inverted relative to the
  true relationship; treat calibrated numbers as hypotheses to confirm
  with more API measurements, not as ground truth.
- **ΔMatch** is the fraction of original fingerprint hashes destroyed.
  1.0 = complete fingerprint destruction on this surrogate. It says
  nothing about what the sponsor's matcher would do until the sandbox API
  is wired in.
- **Quality** metrics (SNR/LSD/spectral-angle) trade off against both
  deltas; the SOW's listening tests and PEAQ/STOI/ViSQOL battery are the
  authoritative quality instruments.

## Notes for the final deliverable

- Per SOW §5, source code and model weights are archived for Sponsor
  review — keep `bench_results/` and the run logs.
- Per SOW §7, AI-assisted code generation used here is logged in the
  report's AI Disclosure Annex.
