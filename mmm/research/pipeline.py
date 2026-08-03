"""
Benchmark pipeline: baseline measurement -> sanitization -> re-measurement.

For each input clip and each sanitizer variant:

1. Baseline: extract features, compute the AI-likeness proxy, compute the
   acoustic fingerprint of the original.
2. Run the real MMM sanitizer (stdout captured to a log file).
3. Re-measure the processed clip: proxy, fingerprint coverage vs the
   original (the ΔMatch surrogate), and objective quality vs the original.

Also computes:
* fingerprint specificity controls (unrelated clips should not match),
* a calibration sanity check (does the proxy separate the synthetic
  sterile vs humanized control classes before any processing?).

Everything runs offline on the local corpus; no live services are called.
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path

import librosa
import numpy as np

from . import features as feat
from . import proxy as prox
from . import fingerprint as fp
from . import quality as qual
from . import corpus as corpus_mod
from .variants import resolve_variants, variant_output_path


def _load_mono(path: Path):
    audio, sr = librosa.load(str(path), sr=None, mono=True)
    audio = np.asarray(audio, dtype=np.float64)
    return audio, sr


def _measure(audio: np.ndarray, sr: int, fingerprint_kwargs: dict) -> dict:
    """Features + proxy + fingerprint for one clip."""
    f = feat.extract_features(audio, sr)
    score = prox.score_ai_likeness(f)
    hashes = fp.spectral_peak_hashes(audio, sr, **fingerprint_kwargs)
    return {
        "features": f,
        "proxy": score,
        "n_hashes": len(hashes),
        "hashes": hashes,
    }


def run_benchmark(
    inputs: list[Path],
    variant_names,
    out_dir: Path,
    seed: int = 42,
    duration: float = 10.0,
    sr: int = corpus_mod.SR_DEFAULT,
    manifest: dict | None = None,
    fingerprint_kwargs: dict | None = None,
) -> dict:
    """
    Run the full offline benchmark.

    Returns a results dict (also written to ``out_dir/results.json``).
    """
    out_dir = Path(out_dir)
    audio_dir = out_dir / "audio"
    log_dir = out_dir / "logs"
    audio_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    fingerprint_kwargs = fingerprint_kwargs or {}
    variants = resolve_variants(variant_names)

    meta = {
        "engagement_ref": "SUB-AHA-2026-SEC-001 (offline phase)",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "note": "Offline sandbox only. No live SubmitHub/AHA/ACRCloud services were contacted.",
        "surrogate_note": (
            "AI-likeness proxy and spectral-peak fingerprint are local surrogates; "
            "the sponsors' sandboxed APIs are the authoritative instruments."
        ),
    }

    manifest = manifest or {}
    class_by_name = {Path(f["path"]).name: f["class"] for f in manifest.get("files", [])}

    files_results = {}
    for input_file in inputs:
        stem = input_file.stem
        print(f"\n=== {stem} ===")
        orig_audio, orig_sr = _load_mono(input_file)
        baseline = _measure(orig_audio, orig_sr, fingerprint_kwargs)
        fname = input_file.name
        file_cls = class_by_name.get(fname, "unlabeled")

        variants_results = {}
        for vname, vcfg in variants:
            log_path = log_dir / f"{stem}__{vname}.log"
            out_path = variant_output_path(vname, input_file, out_dir)
            t0 = time.time()
            row = {
                "variant": vname,
                "desc": vcfg["desc"],
                "success": False,
                "error": None,
                "processing_time_s": None,
            }
            try:
                with open(log_path, "w", encoding="utf-8") as log_fh:
                    with contextlib.redirect_stdout(log_fh), contextlib.redirect_stderr(log_fh):
                        result = vcfg["runner"](input_file, out_path, seed, vcfg["kwargs"])
                row["processing_time_s"] = round(time.time() - t0, 2)
                if result.get("success"):
                    proc_audio, proc_sr = _load_mono(out_path)
                    if proc_sr != orig_sr:
                        proc_audio = librosa.resample(proc_audio, orig_sr=proc_sr, target_sr=orig_sr)
                        proc_sr = orig_sr
                    post = _measure(proc_audio, proc_sr, fingerprint_kwargs)
                    q = qual.quality_metrics(orig_audio, proc_audio, orig_sr)
                    coverage = fp.match_coverage(baseline["hashes"], post["hashes"])

                    row.update(
                        {
                            "success": True,
                            "output_file": str(out_path),
                            "ai_proxy_before": baseline["proxy"]["overall"],
                            "ai_proxy_after": post["proxy"]["overall"],
                            "ai_proxy_delta": round(
                                post["proxy"]["overall"] - baseline["proxy"]["overall"], 2
                            ),
                            "ai_proxy_spectral_before": baseline["proxy"]["spectral"],
                            "ai_proxy_spectral_after": post["proxy"]["spectral"],
                            "ai_proxy_temporal_before": baseline["proxy"]["temporal"],
                            "ai_proxy_temporal_after": post["proxy"]["temporal"],
                            "fingerprint_coverage": round(coverage, 4),
                            "fingerprint_destroyed": round(1.0 - coverage, 4),
                            "n_hashes_before": baseline["n_hashes"],
                            "n_hashes_after": post["n_hashes"],
                            "quality": q,
                            "duration_s": round(len(orig_audio) / orig_sr, 2),
                            "sample_rate": orig_sr,
                            "stats": result.get("stats"),
                        }
                    )
                else:
                    row["error"] = result.get("error", "unknown sanitizer error")
            except Exception as exc:  # noqa: BLE001 — record and continue
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["processing_time_s"] = round(time.time() - t0, 2)

            variants_results[vname] = row
            status = "OK" if row["success"] else f"FAIL({row['error']})"
            print(f"  [{vname:12s}] {status}  ({row['processing_time_s']}s)")

        files_results[fname] = {
            "class": file_cls,
            "baseline": {
                "proxy": baseline["proxy"],
                "n_hashes": baseline["n_hashes"],
                # features stored for downstream analysis (not in the
                # human-readable report)
                "features": baseline["features"],
                "_hashes": baseline["hashes"],  # stripped before serialization
            },
            "variants": variants_results,
        }

    # ---- fingerprint specificity control (unrelated clips must not match)
    controls = []
    names = list(files_results)
    for i in range(min(len(names), 4)):
        a_hashes = files_results[names[i]]["baseline"].get("_hashes")
        if a_hashes is None:
            continue
        for j in range(i + 1, min(len(names), i + 3)):
            b_hashes = files_results[names[j]]["baseline"].get("_hashes")
            if b_hashes is None:
                continue
            controls.append(
                {
                    "a": names[i],
                    "b": names[j],
                    "coverage": round(fp.specificity_check(a_hashes, b_hashes), 4),
                }
            )
    # keep hashes out of the serialized results (they are large)
    for name in names:
        files_results[name]["baseline"].pop("_hashes", None)

    # ---- calibration sanity: does the proxy separate sterile vs humanized?
    calibration = {"sterile_mean": None, "humanized_mean": None, "separation": None, "n_pairs": 0}
    sterile = [
        files_results[n]["baseline"]["proxy"]["overall"]
        for n in names
        if files_results[n]["class"] == "sterile"
    ]
    human = [
        files_results[n]["baseline"]["proxy"]["overall"]
        for n in names
        if files_results[n]["class"] == "humanized"
    ]
    if sterile and human:
        separation = float(np.mean(sterile) - np.mean(human))
        pairs = 0
        for s in sterile:
            for h in human:
                pairs += 1 if s > h else 0
        calibration = {
            "sterile_mean": round(float(np.mean(sterile)), 2),
            "humanized_mean": round(float(np.mean(human)), 2),
            "separation": round(separation, 2),
            "sterile_vs_humanized_pairs_win_rate": round(pairs / (len(sterile) * len(human)), 3),
            "n_pairs": len(sterile) * len(human),
        }

    # ---- summary across variants
    summary = []
    for vname, vcfg in variants:
        rows = [files_results[n]["variants"][vname] for n in names]
        ok = [r for r in rows if r["success"]]
        if not ok:
            summary.append({"variant": vname, "success_rate": 0.0, "n": len(rows)})
            continue
        summary.append(
            {
                "variant": vname,
                "desc": vcfg["desc"],
                "success_rate": round(len(ok) / len(rows), 2),
                "n": len(ok),
                "mean_ai_proxy_delta": round(float(np.mean([r["ai_proxy_delta"] for r in ok])), 2),
                "mean_ai_proxy_spectral_delta": round(
                    float(np.mean([r["ai_proxy_spectral_after"] - r["ai_proxy_spectral_before"] for r in ok])), 2
                ),
                "mean_ai_proxy_temporal_delta": round(
                    float(np.mean([r["ai_proxy_temporal_after"] - r["ai_proxy_temporal_before"] for r in ok])), 2
                ),
                "mean_fingerprint_coverage": round(
                    float(np.mean([r["fingerprint_coverage"] for r in ok])), 4
                ),
                "mean_fingerprint_destroyed": round(
                    float(np.mean([r["fingerprint_destroyed"] for r in ok])), 4
                ),
                "mean_snr_db": round(float(np.mean([r["quality"]["snr_db"] for r in ok])), 2),
                "mean_warp_snr_db": round(
                    float(
                        np.mean(
                            [
                                r["quality"]["warp_snr_db"]
                                for r in ok
                                if r["quality"].get("warp_snr_db") is not None
                            ]
                        )
                    ),
                    2,
                )
                if any(r["quality"].get("warp_snr_db") is not None for r in ok)
                else None,
                "mean_local_aligned_snr_db": round(
                    float(
                        np.mean(
                            [
                                r["quality"]["local_aligned_snr_db"]
                                for r in ok
                                if r["quality"].get("local_aligned_snr_db") is not None
                            ]
                        )
                    ),
                    2,
                )
                if any(r["quality"].get("local_aligned_snr_db") is not None for r in ok)
                else None,
                "mean_aligned_snr_db": round(
                    float(
                        np.mean(
                            [
                                r["quality"]["aligned_snr_db"]
                                for r in ok
                                if r["quality"].get("aligned_snr_db") is not None
                            ]
                        )
                    ),
                    2,
                )
                if any(r["quality"].get("aligned_snr_db") is not None for r in ok)
                else None,
                "mean_lsd_db": round(
                    float(np.mean([r["quality"]["mean_lsd_db"] for r in ok])), 2
                ),
                "mean_spectral_angle": round(
                    float(np.mean([r["quality"]["spectral_angle_cos"] for r in ok])), 4
                ),
                "mean_mfcc_cos": round(
                    float(
                        np.mean(
                            [
                                r["quality"]["mfcc_cos"]
                                for r in ok
                                if r["quality"].get("mfcc_cos") is not None
                            ]
                        )
                    ),
                    4,
                )
                if any(r["quality"].get("mfcc_cos") is not None for r in ok)
                else None,
                "mean_stoi": round(
                    float(np.mean([r["quality"]["stoi"] for r in ok if r["quality"].get("stoi") is not None])), 4
                )
                if any(r["quality"].get("stoi") is not None for r in ok)
                else None,
                "mean_processing_time_s": round(
                    float(np.mean([r["processing_time_s"] for r in ok])), 2
                ),
            }
        )

    results = {
        "meta": meta,
        "corpus": {
            "inputs": [str(p) for p in inputs],
            "manifest": manifest,
        },
        "calibration": calibration,
        "fingerprint_controls": controls,
        "summary": summary,
        "files": files_results,
    }

    with open(out_dir / "results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)

    return results
