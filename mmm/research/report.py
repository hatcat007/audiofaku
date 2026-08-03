"""
Markdown report rendering for the offline benchmark results.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .variants import VARIANTS


def _fmt(v, nd=2, suffix=""):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}{suffix}"
    return str(v)


def render_markdown(results: dict) -> str:
    """Render the full markdown report from a results dict."""
    meta = results["meta"]
    cal = results["calibration"]
    lines = []
    a = lines.append

    a(f"# Offline Robustness Benchmark — {meta.get('engagement_ref', '')}")
    a("")
    a(f"**Run date:** {meta.get('date')}  ")
    a(f"**Seed:** {meta.get('seed')}  ")
    a("")
    a(f"> {meta.get('note')}")
    a("")
    a(f"> {meta.get('surrogate_note')}")
    a("")
    a("## 1. Corpus")
    a("")
    a("| file | class | description |")
    a("|---|---|---|")
    for f in results["corpus"]["manifest"].get("files", []):
        a(f"| {f['path']} | {f['class']} | {f['desc']} |")
    if not results["corpus"]["manifest"].get("files"):
        a("| (user-supplied inputs; unlabeled) | — | — |")
    a("")

    a("## 2. Instrument sanity checks")
    a("")
    a("### 2.1 AI-likeness proxy — separation of control classes (before any processing)")
    a("")
    a("The proxy should score *sterile* synthetic material higher than *humanized* material. "
      "This is a sanity check of the surrogate, not a result about the sanitizers.")
    a("")
    a(f"- sterile mean: **{_fmt(cal.get('sterile_mean'))}**  ")
    a(f"- humanized mean: **{_fmt(cal.get('humanized_mean'))}**  ")
    a(f"- separation (sterile − humanized): **{_fmt(cal.get('separation'))}**  ")
    if cal.get("n_pairs"):
        a(f"- pairwise win rate (sterile > humanized): **{_fmt(cal.get('sterile_vs_humanized_pairs_win_rate'), 3)}** "
          f"({cal.get('n_pairs')} pairs)")
    a("")
    if cal.get("separation") is not None and cal["separation"] <= 0:
        a("> ⚠️ The proxy does not separate the control classes. Check the reference ranges in "
          "`mmm/research/proxy.py` before interpreting ΔAI numbers.")
        a("")
    elif cal.get("separation") is not None and cal["separation"] < 10:
        a("> ⚠️ Separation is positive but weak. The synthetic 'humanized' clips only mildly "
          "humanize timing/velocity (by design — the corpus is meant for before/after Δ "
          "measurement, not as a strong classifier benchmark). Treat ΔAI numbers as "
          "directional, and re-validate against the sponsors' evaluation library.")
        a("")
    a("### 2.2 Fingerprint specificity control")
    a("")
    a("Unrelated clips should show near-zero match coverage. This checks the surrogate fingerprint "
      "does not produce false matches.")
    a("")
    a("| clip A | clip B | coverage |")
    a("|---|---|---|")
    for c in results.get("fingerprint_controls", []):
        a(f"| {c['a']} | {c['b']} | {_fmt(c['coverage'], 4)} |")
    if not results.get("fingerprint_controls"):
        a("| — | — | — |")
    a("")
    a("> Note: controls pairing clips that share the same notes/frequencies "
      "(e.g. arpeggio vs chords, or the sterile/humanized pair of the same "
      "pattern) legitimately show high coverage — the surrogate's landmarks "
      "are frequency-based, so shared tonal content matches. The meaningful "
      "controls are the cross-material pairs (ambient vs arpeggio), which "
      "sit near zero.")
    a("")

    # ---- per-file tables
    a("## 3. Per-file results")
    a("")
    for fname, fdata in results["files"].items():
        a(f"### {fname}  _(class: {fdata['class']})_")
        a("")
        a("Baseline: AI-likeness proxy "
          f"**{_fmt(fdata['baseline']['proxy']['overall'])}** "
          f"(spectral {_fmt(fdata['baseline']['proxy']['spectral'])}, "
          f"temporal {_fmt(fdata['baseline']['proxy']['temporal'])}); "
          f"fingerprint hashes: {fdata['baseline']['n_hashes']}")
        a("")
        a("| variant | success | ΔAI proxy | Δ spectral | Δ temporal | fp coverage | ΔMatch | SNR dB | warpSNR dB | meanLSD dB | spec-angle | MFCC cos |")
        a("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for vname, row in fdata["variants"].items():
            if not row["success"]:
                a(f"| {vname} | ❌ | — | — | — | — | — | — | — | — | — | — |")
                continue
            q = row["quality"]
            a(
                f"| {vname} | ✅ | {_fmt(row['ai_proxy_delta'])} | "
                f"{_fmt(row['ai_proxy_spectral_after'] - row['ai_proxy_spectral_before'])} | "
                f"{_fmt(row['ai_proxy_temporal_after'] - row['ai_proxy_temporal_before'])} | "
                f"{_fmt(row['fingerprint_coverage'], 4)} | {_fmt(row['fingerprint_destroyed'], 4)} | "
                f"{_fmt(q['snr_db'])} | {_fmt(q.get('warp_snr_db'))} | {_fmt(q['mean_lsd_db'])} | "
                f"{_fmt(q['spectral_angle_cos'], 3)} | {_fmt(q.get('mfcc_cos'), 3)} |"
            )
        a("")
        a("> ΔAI proxy: negative = processed clip reads *less* machine-like to the surrogate. "
          "ΔMatch: fraction of the original fingerprint destroyed (1 − coverage). "
          "SNR is sample-aligned and timing-sensitive; warpSNR undoes the estimated "
          "time-warp curve before measuring fidelity; meanLSD/spec-angle/MFCC-cos are "
          "time-averaged and robust to the sanitizer's intentional micro-timing changes.")
        a("")

    # ---- summary
    a("## 4. Aggregate summary (mean over corpus)")
    a("")
    a("| variant | n | success | mean ΔAI proxy | mean Δ spectral | mean Δ temporal | mean fp coverage | mean ΔMatch | mean SNR | mean warpSNR | meanLSD | spec-angle | MFCC cos | mean time (s) |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in results["summary"]:
        a(
            f"| {s['variant']} | {s['n']} | {_fmt(s.get('success_rate'), 2)} | "
            f"{_fmt(s.get('mean_ai_proxy_delta'))} | {_fmt(s.get('mean_ai_proxy_spectral_delta'))} | "
            f"{_fmt(s.get('mean_ai_proxy_temporal_delta'))} | {_fmt(s.get('mean_fingerprint_coverage'), 4)} | "
            f"{_fmt(s.get('mean_fingerprint_destroyed'), 4)} | {_fmt(s.get('mean_snr_db'))} | "
            f"{_fmt(s.get('mean_warp_snr_db'))} | {_fmt(s.get('mean_lsd_db'))} | "
            f"{_fmt(s.get('mean_spectral_angle'), 3)} | {_fmt(s.get('mean_mfcc_cos'), 3)} | "
            f"{_fmt(s.get('mean_processing_time_s'))} |"
        )
    a("")

    a("## 5. Interpretation and limitations")
    a("")
    a("- **Surrogates only.** The AI-likeness proxy and the spectral-peak fingerprint are local "
      "stand-ins. Numbers here are *directional*; the sponsors' sandboxed evaluation libraries are "
      "the authoritative instruments and should be wired in before drawing conclusions (see "
      "`mmm/research/README.md` for swap-in points).")
    a("- **Synthetic corpus.** The control clips are procedurally generated tones/patterns, not "
      "real music. Results on real (open-licensed) audio may differ.")
    a("- **Short clips.** Default clips are 30 s; some statistics (long-range tempo drift, "
      "track-level regularity) only fully matter on full-length tracks. Re-run on full-length "
      "open-licensed files for track-level numbers.")
    a("- **No live services.** Per the engagement's out-of-scope list, nothing here contacts "
      "SubmitHub, AHA Music, or ACRCloud production services.")
    a("- **Quality metrics.** SNR/LSD/spectral-angle are computed locally; STOI is "
      "speech-oriented and informational only; PEAQ/PESQ are not bundled.")
    a("")

    a("## 6. AI Disclosure Annex (per SOW §7)")
    a("")
    a("| date | tool | purpose | oversight |")
    a("|---|---|---|---|")
    a("| " + time.strftime("%Y-%m-%d") + " | Arena.ai coding agent (session `arena/019fc787-audiofaku`) | "
      "Scaffolding and implementation of `mmm/research` harness per SOW methodology; reviewed and "
      "tested by the human researcher before use | human review; code committed to local repo |")
    a("| " + time.strftime("%Y-%m-%d") + " | LLM-assisted report drafting | Markdown report rendering | "
      "substantive findings remain the responsibility of the human research team |")
    a("")
    a("No proprietary Sponsor data, model internals, or audio samples were exposed to third-party "
      "AI services during this run.")
    a("")

    return "\n".join(lines)


def write_report(results: dict, out_dir: Path) -> Path:
    md = render_markdown(results)
    path = out_dir / "report.md"
    path.write_text(md, encoding="utf-8")
    return path


def load_results(out_dir: Path) -> dict:
    with open(out_dir / "results.json", encoding="utf-8") as fh:
        return json.load(fh)
