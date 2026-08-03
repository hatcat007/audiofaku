#!/usr/bin/env python3
"""
mmm.research — offline robustness benchmark CLI.

Usage examples
--------------
    # Generate the synthetic corpus and run the default variants:
    python -m mmm.research

    # Run on your own open-licensed files:
    python -m mmm.research --inputs /path/to/track.wav /path/to/dir

    # Run all documented sanitizer variants:
    python -m mmm.research --variants all

    # Longer clips, different output dir:
    python -m mmm.research --duration 20 --out bench_results/run2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import corpus as corpus_mod
from .pipeline import run_benchmark
from .report import write_report
from .variants import VARIANTS, resolve_variants


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mmm.research",
        description="Offline robustness benchmark (SUB-AHA-2026-SEC-001, offline phase).",
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help="Input audio files/directories (wav/mp3/flac/ogg). If omitted, a synthetic "
        "control corpus is generated.",
    )
    parser.add_argument(
        "--variants",
        default=",".join(["preserving", "paranoid"]),
        help=f"Comma-separated sanitizer variants, or 'all'. "
        f"Available: {', '.join(VARIANTS)}",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Corpus clip duration in seconds (default 30; shorter clips run faster but "
        "the tempo-drift stage is designed for track-length material, so use >= 30 for "
        "representative numbers)",
    )
    parser.add_argument("--sr", type=int, default=corpus_mod.SR_DEFAULT, help="Corpus sample rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--out",
        default="bench_results",
        help="Output directory (default: bench_results)",
    )
    parser.add_argument(
        "--no-corpus",
        action="store_true",
        help="Do not generate a corpus when --inputs is empty (fail instead)",
    )
    args = parser.parse_args(argv)

    try:
        resolve_variants(args.variants)
    except ValueError as exc:
        parser.error(str(exc))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.inputs:
        inputs, manifest = corpus_mod.discover_inputs(args.inputs, out_dir / "corpus")
    else:
        if args.no_corpus:
            parser.error("--no-corpus requires --inputs")
        print(f"Generating synthetic corpus ({args.duration}s @ {args.sr} Hz, seed={args.seed}) ...")
        inputs, manifest = corpus_mod.discover_inputs(None, out_dir / "corpus")
        # regenerate at requested duration/sr if different from existing manifest
        if (
            manifest.get("duration_seconds") != args.duration
            or manifest.get("sr") != args.sr
            or manifest.get("seed") != args.seed
        ):
            manifest = corpus_mod.generate_corpus(
                out_dir / "corpus", duration=args.duration, sr=args.sr, seed=args.seed
            )
            inputs = sorted(out_dir / "corpus" / f["path"] for f in manifest["files"])

    if not inputs:
        parser.error("no input audio files found")

    print(f"Inputs ({len(inputs)}):")
    for p in inputs:
        print(f"  - {p}")

    results = run_benchmark(
        inputs=inputs,
        variant_names=args.variants,
        out_dir=out_dir,
        seed=args.seed,
        duration=args.duration,
        sr=args.sr,
        manifest=manifest,
    )
    report_path = write_report(results, out_dir)

    print("\n" + "=" * 78)
    print("SUMMARY (means over corpus)")
    print("=" * 78)
    for s in results["summary"]:
        ok = "OK" if s.get("success_rate", 0) > 0 else "FAIL"
        print(
            f"  [{s['variant']:12s}] {ok}  ΔAI {s.get('mean_ai_proxy_delta', '—'):>7}  "
            f"ΔMatch {s.get('mean_fingerprint_destroyed', '—'):>7}  "
            f"SNR {s.get('mean_snr_db', '—'):>6} dB  warpSNR {s.get('mean_warp_snr_db', '—'):>6} dB  "
            f"meanLSD {s.get('mean_lsd_db', '—'):>6} dB  ({s.get('mean_processing_time_s', '—')} s/file)"
        )
    print("=" * 78)
    print(f"Report:  {report_path}")
    print(f"JSON:    {out_dir / 'results.json'}")
    print(f"Audio:   {out_dir / 'audio'}")
    print(f"Logs:    {out_dir / 'logs'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
