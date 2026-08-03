"""
Stage-ablation runner: apply ONE sanitizer stage at a time to a track and
measure the calibrated proxy, fingerprint coverage, and quality.

Purpose (SOW §3/§4): identify which individual processing stages move
which detector sub-scores (spectral vs temporal) and at what quality cost.
Runs fully offline against the local surrogates.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import librosa
import numpy as np

from . import fingerprint as fp
from . import proxy_v2 as prox
from . import quality as qual
from .shlabs import extract_sh_features
from .features import extract_features


def _load_mono(path: Path):
    audio, sr = librosa.load(str(path), sr=None, mono=True)
    return np.asarray(audio, dtype=np.float64), sr


def _features(audio: np.ndarray, sr: int) -> dict:
    f = extract_features(audio, sr)
    f.update(extract_sh_features(audio, sr))
    return f


STAGES = {
    "metadata_only": {
        "desc": "mutagen tag strip only (no audio processing)",
        "fn": "meta",
    },
    "spectral_clean": {"desc": "spectral watermark cleaning", "fn": "spectral_clean"},
    "fingerprint_removal": {"desc": "statistical fingerprint removal", "fn": "fingerprint"},
    "hf_noise": {"desc": "HF noise + dither", "fn": "hf_noise"},
    "humanization": {"desc": "wow/flutter + gain swings + decorrelation", "fn": "humanization"},
    "micro_warp": {"desc": "micro resample warp", "fn": "micro_warp"},
    "analog_warmth": {"desc": "soft saturation + DC block", "fn": "analog"},
    "micro_ambience": {"desc": "crossfeed delay + all-pass", "fn": "ambience"},
    "bandlimit": {"desc": "gentle low-pass (19-20 kHz)", "fn": "bandlimit"},
    "resample_nudge": {"desc": "resample nudge", "fn": "resample_nudge"},
    "phase_swirl": {"desc": "all-pass phase swirl", "fn": "phase_swirl"},
    "phase_noise": {"desc": "FFT phase noise", "fn": "phase_noise"},
    "micro_eq": {"desc": "micro EQ modulation", "fn": "micro_eq"},
    "comb_mask": {"desc": "dynamic comb mask", "fn": "comb_mask"},
    "transient_shift": {"desc": "transient micro-shift", "fn": "transient"},
    "onset_velocity": {"desc": "onset velocity variation", "fn": "onset_velocity"},
    "tempo_drift": {"desc": "long-range tempo drift", "fn": "tempo_drift"},
    "mfcc_perturb": {"desc": "MFCC-targeted perturbation", "fn": "mfcc"},
    "clarity_tilt": {"desc": "high-shelf clarity tilt", "fn": "clarity_tilt"},
}


def apply_stage(stage: str, audio: np.ndarray, sr: int, seed: int) -> np.ndarray:
    """Apply one stage to audio (samples, channels).  Returns processed audio."""
    from mmm import preserving_sanitizer as ps

    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    fn = STAGES[stage]["fn"]
    if fn == "meta":
        return audio.copy()
    if fn == "spectral_clean":
        return ps._apply_spectral_watermark_cleaning(audio, sr, False)
    if fn == "fingerprint":
        return ps._apply_fingerprint_removal(audio, sr, False)
    if fn == "hf_noise":
        return ps._add_hf_noise_and_dither(audio, sr, False)
    if fn == "humanization":
        return ps._apply_humanization(audio, sr, False)
    if fn == "micro_warp":
        return ps._apply_micro_resample_warp(audio, sr, False)
    if fn == "analog":
        return ps._apply_analog_warmth(audio, sr, False)
    if fn == "ambience":
        return ps._add_micro_ambience(audio, sr, False)
    if fn == "bandlimit":
        return ps._apply_gentle_bandlimit(audio, sr, False)
    if fn == "resample_nudge":
        return ps._apply_resample_nudge(audio, sr, False)
    if fn == "phase_swirl":
        return ps._apply_phase_swirl(audio, sr, False)
    if fn == "phase_noise":
        return ps._apply_phase_noise_fft(audio, False)
    if fn == "micro_eq":
        return ps._apply_micro_eq_modulation(audio, sr, False)
    if fn == "comb_mask":
        return ps._apply_dynamic_comb_mask(audio, sr, False)
    if fn == "transient":
        return ps._apply_transient_micro_shift(audio, sr, False)
    if fn == "onset_velocity":
        return ps._apply_onset_velocity_variation(audio, sr, False)
    if fn == "tempo_drift":
        return ps._apply_long_range_tempo_drift(audio, sr, False)
    if fn == "mfcc":
        return ps._apply_mfcc_perturbation(audio, sr, False)
    if fn == "clarity_tilt":
        return ps._apply_clarity_tilt(audio, sr, False)
    raise ValueError(f"unknown stage {stage}")


def run_ablation(
    input_file: Path,
    out_dir: Path,
    stages: list[str] | None = None,
    seed: int = 42,
    max_hashes: int = 1_500_000,
    resume: bool = False,
) -> dict:
    out_dir = Path(out_dir)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    if stages is None:
        stages = list(STAGES)
    stages = [s for s in stages if s in STAGES]

    orig_audio, orig_sr = _load_mono(input_file)
    orig_feats = _features(orig_audio, orig_sr)
    base_score = prox.score_ai_likeness(orig_feats)
    base_hashes = fp.spectral_peak_hashes(
        orig_audio, orig_sr, max_hashes=max_hashes
    )
    gc.collect()

    results: dict
    results_path = out_dir / "ablation.json"
    if resume and results_path.exists():
        with open(results_path, encoding="utf-8") as fh:
            results = json.load(fh)
        # replace baseline with fresh measurement
        results["meta"]["baseline"] = base_score
    else:
        results = {
            "meta": {
                "engagement_ref": "SUB-AHA-2026-SEC-001 (offline phase)",
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "seed": seed,
                "input": str(input_file),
                "baseline": base_score,
                "note": "Stage ablation, offline surrogates only.",
            },
            "stages": {},
        }

    for stage in stages:
        if resume and stage in results["stages"]:
            print(f"  [{stage:18s}] skipped (already measured)")
            continue
        t0 = time.time()
        row = {"desc": STAGES[stage]["desc"], "success": False, "error": None}
        try:
            out_path = audio_dir / f"{stage}.wav"
            if not out_path.exists():
                # load stereo for the sanitizer stages, mono for measurement
                audio_st, sr_st = librosa.load(str(input_file), sr=None, mono=False)
                audio_st = np.asarray(audio_st, dtype=np.float64)
                if audio_st.ndim == 1:
                    audio_st = audio_st.reshape(-1, 1)
                elif audio_st.shape[0] < audio_st.shape[1]:
                    audio_st = np.ascontiguousarray(audio_st.T)
                out_audio = apply_stage(stage, audio_st, sr_st, seed)

                # save stereo WAV
                peak = np.max(np.abs(out_audio)) or 1.0
                out16 = np.clip(out_audio / peak * 0.95, -1, 1)
                import soundfile as sf

                sf.write(out_path, out16, sr_st, subtype="PCM_16")
            else:
                out_audio, sr_st = librosa.load(str(out_path), sr=None, mono=False)
                out_audio = np.asarray(out_audio, dtype=np.float64)
                if out_audio.ndim == 1:
                    out_audio = out_audio.reshape(-1, 1)
                elif out_audio.shape[0] < out_audio.shape[1]:
                    out_audio = np.ascontiguousarray(out_audio.T)

            # measure mono
            mono = out_audio.mean(axis=1)
            feats = _features(mono, sr_st)
            score = prox.score_ai_likeness(feats)
            proc_hashes = fp.spectral_peak_hashes(mono, sr_st, max_hashes=max_hashes)
            coverage = fp.match_coverage(base_hashes, proc_hashes)
            q = qual.quality_metrics(orig_audio, mono, orig_sr)
            gc.collect()

            row.update(
                {
                    "success": True,
                    "processing_time_s": round(time.time() - t0, 1),
                    "proxy_before": base_score,
                    "proxy_after": score,
                    "d_overall": round(score["overall"] - base_score["overall"], 2),
                    "d_spectral": round(score["spectral"] - base_score["spectral"], 2),
                    "d_temporal": round(score["temporal"] - base_score["temporal"], 2),
                    "d_raw_spectral": round(
                        score["raw_spectral"] - base_score["raw_spectral"], 2
                    ),
                    "d_raw_temporal": round(
                        score["raw_temporal"] - base_score["raw_temporal"], 2
                    ),
                    "fingerprint_coverage": round(coverage, 4),
                    "fingerprint_destroyed": round(1.0 - coverage, 4),
                    "quality": {
                        "snr_db": q["snr_db"],
                        "warp_snr_db": q.get("warp_snr_db"),
                        "mean_lsd_db": q.get("mean_lsd_db"),
                        "spectral_angle": q["spectral_angle_cos"],
                        "mfcc_cos": q.get("mfcc_cos"),
                    },
                }
            )
            print(
                f"  [{stage:18s}] OK  dSpec={row['d_raw_spectral']:6.1f} "
                f"dTemp={row['d_raw_temporal']:6.1f} "
                f"dMatch={row['fingerprint_destroyed']:.3f} "
                f"warpSNR={q.get('warp_snr_db')} ({row['processing_time_s']}s)"
            )
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["processing_time_s"] = round(time.time() - t0, 1)
            print(f"  [{stage:18s}] FAIL {exc}")

        results["stages"][stage] = row
        # save incrementally so a long run can be resumed after a crash
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)

    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mmm.research.ablation")
    ap.add_argument("--input", required=True, help="input audio file")
    ap.add_argument("--out", default="bench_results/ablation", help="output dir")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--stages",
        default=None,
        help="comma-separated stage subset (default: all)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="resume: keep existing ablation.json entries and stage audio",
    )
    args = ap.parse_args(argv)

    stages = [s.strip() for s in args.stages.split(",")] if args.stages else None
    res = run_ablation(
        Path(args.input), Path(args.out), stages, seed=args.seed, resume=args.resume
    )

    # summary table
    print("\n" + "=" * 90)
    print("STAGE ABLATION SUMMARY (deltas vs unprocessed original)")
    print("=" * 90)
    print(f"{'stage':18s} {'dRawSpec':>8s} {'dRawTemp':>8s} {'dSpecCal':>8s} {'dTempCal':>8s} {'ΔMatch':>7s} {'warpSNR':>8s} {'mlSD':>6s} {'time':>6s}")
    for s, r in res["stages"].items():
        if not r["success"]:
            print(f"{s:18s} FAILED")
            continue
        q = r["quality"]
        print(
            f"{s:18s} {r['d_raw_spectral']:8.1f} {r['d_raw_temporal']:8.1f} "
            f"{r['d_spectral']:8.1f} {r['d_temporal']:8.1f} {r['fingerprint_destroyed']:7.3f} "
            f"{q.get('warp_snr_db') if q.get('warp_snr_db') is not None else -99:8.1f} "
            f"{q.get('mean_lsd_db') if q.get('mean_lsd_db') is not None else -99:6.1f} "
            f"{r['processing_time_s']:6.1f}"
        )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
