# Robustness Assessment Brief — Audio Identification & AI-Content Detection

**Engagement ref:** SUB-AHA-2026-SEC-001 (offline/measured phase)
**Track under test:** `tests/song1.mp3` — "El Próximo Verano" (Suno-generated, 3:29, 48 kHz stereo)
**Date:** 2026-08-04
**Status:** Draft for mid-term report (SOW §5)

---

## 1. Scope of this brief

This brief summarizes measured results from the offline robustness assessment:
8 real SubmitHub AI Song Checker measurements on the target track across
6 sanitizer variants and 4 single-stage ablations, plus the local surrogate
harness (`mmm/research/`). All checker measurements were provided by the
researcher via the internal API; the tooling itself ran fully offline and
contacted no live production service.

## 2. Measured results (real checker, Pure-AI %)

| file | spectral | temporal | verdict driver |
|---|---|---|---|
| 06_fast (metadata-only) | 87 | 92 | spectral |
| 02_preserving (default flags) | 87 | **19** | spectral |
| 03_paranoid | 86 | 32 | spectral |
| 05_stealth_plus | 84 | 44 | spectral |
| ablation_tempo_drift | 90 | 34 | spectral |
| ablation_humanization | 89 | 78 | spectral |
| ablation_spectral_clean | 89 | 88 | spectral |
| ablation_transient_shift | 89 | 87 | spectral |

Spectral range across all processing: **84–90 (mean 87.6)**.
Temporal range: **19–92** (fully controllable).

## 3. Findings

### F1 — Metadata/provenance removal is complete but irrelevant to detection
All variants strip the C2PA provenance manifest, "made with suno" tags, the
Suno song-ID link, lyrics, and cover art (verified: `tags: NONE` on outputs).
The metadata-only variant scores 87/92 — indistinguishable from unprocessed.
Detection signal lives entirely in the audio content.

### F2 — The spectral classifier is invariant to the entire DSP family tested
Six processing stacks, four single stages — spectral never left 84–90.
Every hand-crafted spectral feature is *anti-correlated* with the real
spectral score (Spearman −0.26…−0.79 on 8 anchors). Consistent with the
v4 methodology (spectral = "turning each song into a picture"): the
classifier keys on spectrogram-image texture that content-preserving
transforms (phase noise, EQ, HF cleaning, resampling, band-limiting) do
not alter.

### F3 — Temporal is controllable, and the full stack is super-additive
- Default `preserving` reaches **temporal 19** — the lowest measured value,
  beating every single stage (best single: tempo_drift at 34) and both
  heavier variants (paranoid 32, stealth_plus 44).
- Extra stealth flags are counterproductive for temporal on this track.
- Single-stage deltas are modest (92 → 34–88); the default full stack
  (→ 19) exceeds the sum of its parts.

### F4 — Offline prediction of within-processed temporal is not possible with hand-crafted features
Leave-one-out MAE of the best feature fit ≈ 29 over the 8 anchors.
`beat_cv` is the only individually significant feature (Spearman ρ=0.88,
p=0.004) but insufficient for ranking. tempo_drift is invisible to all
features tested (beat CV, RMS dynamics, tempo stability, beat-phase drift;
fine-grained tempo estimator noise floor ~17 BPM swamps the ~1 BPM drift).
**Implication:** within-processed ranking must be measured with the real
API; offline surrogates are coarse discriminators (untouched vs processed)
only.

### F5 — Quality trade-off is quantified
- preserving-family variants: spectral angle ≥ 0.998, MFCC cosine ≥ 0.98,
  masked mean-LSD 6–6.8 dB — tonal content preserved.
- Sample-level SNR ≈ −3 dB across variants: reflects intentional
  micro-timing/phase changes (tempo drift, transient shifts), not content
  loss; warp-compensated SNR ≈ 0–0.5 dB.
- `fast` is near-lossless (SNR 74 dB) but only strips metadata — no
  detection effect.
- Listening: all preserving-family variants are audibly similar to the
  original; heavier stacks (paranoid/max) show subtle HF smoothing.

### F6 — Feasibility conclusion for this track
**DSP-only, quality-preserving sanitization cannot move the spectral
classifier; an undisclosed "pass" for this track is not achievable within
that constraint.** Content-level changes (stems, re-production,
resynthesis) would be required, which change the nature of the work and
raise disclosure obligations. The honest, reportable result is: temporal
detection is defeatable; spectral detection is robust to this attack class.

## 4. Defensive recommendations (for the sponsors)

1. **Spectral texture is the robust signal — keep it.** The spectrogram-
   image classifier resisted every content-preserving transform tested.
   Harden around it; do not rely on temporal features alone.
2. **Temporal features are attackable and super-additive.** Add redundancy:
   multiple independent temporal cues (beat regularity, onset jitter,
   phase coherence, long-range section correlation) and ensemble over
   long windows; single-cue temporal models are easily pushed down.
3. **Drift-robust tempo/phase features are missing.** The tempo-drift
   attack (strongest single temporal reducer, 92→34) is invisible to
   standard tempo estimators. A drift-aware feature (e.g., beat-phase
   offset slope over time) would close a real gap.
4. **Platform attribution is brittle.** Labels shifted suno_55 ↔ suno_5
   under processing; attribution should not gate decisions.
5. **Metadata removal is a solved problem for attackers — do not treat
   it as a control.** C2PA + ID3 stripping is trivial and changes nothing.

## 5. Open items / next steps

- [ ] Wire in the sponsors' sandboxed evaluation library as the
      authoritative instrument (swap-in points documented in
      `mmm/research/README.md`).
- [ ] Validate on a second Suno track and one open-licensed human track
      (control).
- [ ] MUSHRA-style listening tests (SOW §4) on preserving vs original.
- [ ] Codec-transcoding and purification experiments as documented,
      disclosed measurements (SOW §3 in-scope techniques).

## 6. AI disclosure annex (SOW §7)

| date | tool | purpose | oversight |
|---|---|---|---|
| 2026-08-03/04 | Arena.ai coding agent (session `arena/019fc787-audiofaku`) | Offline harness, feature extraction, calibration, ablation, this brief | human review; code committed to local repo; no live service calls |

No proprietary Sponsor data, model internals, or audio samples were
exposed to third-party AI services. Checker results were entered manually
from researcher reports of the internal API.

---

*Prepared for the robustness-assessment deliverable. Findings are measured
on one track and should not be generalized without the validation items in
§5.*
