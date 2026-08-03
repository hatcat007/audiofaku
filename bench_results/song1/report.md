# Offline Robustness Benchmark — SUB-AHA-2026-SEC-001 (offline phase)

**Run date:** 2026-08-03 17:22:42  
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
| max | ✅ | -0.68 | 6.49 | -7.86 | 0.5937 | 0.4063 | -2.99 | -0.45 | 6.45 | 0.998 | 0.984 |

> ΔAI proxy: negative = processed clip reads *less* machine-like to the surrogate. ΔMatch: fraction of the original fingerprint destroyed (1 − coverage). SNR is sample-aligned and timing-sensitive; warpSNR undoes the estimated time-warp curve before measuring fidelity; meanLSD/spec-angle/MFCC-cos are time-averaged and robust to the sanitizer's intentional micro-timing changes.

## 4. Aggregate summary (mean over corpus)

| variant | n | success | mean ΔAI proxy | mean Δ spectral | mean Δ temporal | mean fp coverage | mean ΔMatch | mean SNR | mean warpSNR | meanLSD | spec-angle | MFCC cos | mean time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| max | 1 | 1.00 | -0.68 | 6.49 | -7.86 | 0.5937 | 0.4063 | -2.99 | -0.45 | 6.45 | 0.998 | 0.984 | 342.93 |

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
