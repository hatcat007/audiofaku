# Offline Robustness Benchmark — SUB-AHA-2026-SEC-001 (offline phase)

**Run date:** 2026-08-03 15:13:53  
**Seed:** 42  

> Offline sandbox only. No live SubmitHub/AHA/ACRCloud services were contacted.

> AI-likeness proxy and spectral-peak fingerprint are local surrogates; the sponsors' sandboxed APIs are the authoritative instruments.

## 1. Corpus

| file | class | description |
|---|---|---|
| (user-supplied inputs; unlabeled) | — | — |

## 2. Instrument sanity checks

### 2.1 AI-likeness proxy — separation of control classes (before any processing)

The proxy should score *sterile* synthetic material higher than *humanized* material. This is a sanity check of the surrogate, not a result about the sanitizers.

- sterile mean: **—**  
- humanized mean: **—**  
- separation (sterile − humanized): **—**  

### 2.2 Fingerprint specificity control

Unrelated clips should show near-zero match coverage. This checks the surrogate fingerprint does not produce false matches.

| clip A | clip B | coverage |
|---|---|---|
| — | — | — |

> Note: controls pairing clips that share the same notes/frequencies (e.g. arpeggio vs chords, or the sterile/humanized pair of the same pattern) legitimately show high coverage — the surrogate's landmarks are frequency-based, so shared tonal content matches. The meaningful controls are the cross-material pairs (ambient vs arpeggio), which sit near zero.

## 3. Per-file results

### song1.wav  _(class: unlabeled)_

Baseline: AI-likeness proxy **43.81** (spectral 33.86, temporal 53.76); fingerprint hashes: 309430

| variant | success | ΔAI proxy | Δ spectral | Δ temporal | fp coverage | ΔMatch | SNR dB | warpSNR dB | meanLSD dB | spec-angle | MFCC cos |
|---|---|---|---|---|---|---|---|---|---|---|---|
| preserving | ✅ | 1.00 | 6.88 | -4.88 | 0.6736 | 0.3264 | -3.01 | 0.51 | 6.77 | 0.998 | 0.987 |
| paranoid | ✅ | -0.53 | 7.02 | -8.07 | 0.5874 | 0.4126 | -2.88 | -0.37 | 6.31 | 0.999 | 0.984 |
| max | ✅ | -0.67 | 6.49 | -7.83 | 0.5934 | 0.4066 | -2.99 | -0.46 | 6.45 | 0.998 | 0.984 |
| stealth_plus | ✅ | 2.91 | 7.11 | -1.28 | 0.5792 | 0.4208 | -2.93 | -0.21 | 5.95 | 0.999 | 0.984 |
| fast | ✅ | -0.02 | -0.04 | 0.00 | 0.9971 | 0.0029 | 74.01 | 74.01 | 0.11 | 1.000 | 1.000 |

> ΔAI proxy: negative = processed clip reads *less* machine-like to the surrogate. ΔMatch: fraction of the original fingerprint destroyed (1 − coverage). SNR is sample-aligned and timing-sensitive; warpSNR undoes the estimated time-warp curve before measuring fidelity; meanLSD/spec-angle/MFCC-cos are time-averaged and robust to the sanitizer's intentional micro-timing changes.

## 4. Aggregate summary (mean over corpus)

| variant | n | success | mean ΔAI proxy | mean Δ spectral | mean Δ temporal | mean fp coverage | mean ΔMatch | mean SNR | mean warpSNR | meanLSD | spec-angle | MFCC cos | mean time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| preserving | 1 | 1.00 | 1.00 | 6.88 | -4.88 | 0.6736 | 0.3264 | -3.01 | 0.51 | 6.77 | 0.998 | 0.987 | 211.40 |
| paranoid | 1 | 1.00 | -0.53 | 7.02 | -8.07 | 0.5874 | 0.4126 | -2.88 | -0.37 | 6.31 | 0.999 | 0.984 | 313.89 |
| max | 1 | 1.00 | -0.67 | 6.49 | -7.83 | 0.5934 | 0.4066 | -2.99 | -0.46 | 6.45 | 0.998 | 0.984 | 334.66 |
| stealth_plus | 1 | 1.00 | 2.91 | 7.11 | -1.28 | 0.5792 | 0.4208 | -2.93 | -0.21 | 5.95 | 0.999 | 0.984 | 320.40 |
| fast | 1 | 1.00 | -0.02 | -0.04 | 0.00 | 0.9971 | 0.0029 | 74.01 | 74.01 | 0.11 | 1.000 | 1.000 | 0.91 |

## 5. Interpretation and limitations

- **Surrogates only.** The AI-likeness proxy and the spectral-peak fingerprint are local stand-ins. Numbers here are *directional*; the sponsors' sandboxed evaluation libraries are the authoritative instruments and should be wired in before drawing conclusions (see `mmm/research/README.md` for swap-in points).
- **Synthetic corpus.** The control clips are procedurally generated tones/patterns, not real music. Results on real (open-licensed) audio may differ.
- **Short clips.** Default clips are 30 s; some statistics (long-range tempo drift, track-level regularity) only fully matter on full-length tracks. Re-run on full-length open-licensed files for track-level numbers.
- **No live services.** Per the engagement's out-of-scope list, nothing here contacts SubmitHub, AHA Music, or ACRCloud production services.
- **Quality metrics.** SNR/LSD/spectral-angle are computed locally; STOI is speech-oriented and informational only; PEAQ/PESQ are not bundled.

## 6. AI Disclosure Annex (per SOW §7)

| date | tool | purpose | oversight |
|---|---|---|---|
| 2026-08-03 | Arena.ai coding agent (session `arena/019fc787-audiofaku`) | Scaffolding and implementation of `mmm/research` harness per SOW methodology; reviewed and tested by the human researcher before use | human review; code committed to local repo |
| 2026-08-03 | LLM-assisted report drafting | Markdown report rendering | substantive findings remain the responsibility of the human research team |

No proprietary Sponsor data, model internals, or audio samples were exposed to third-party AI services during this run.
