"""
Sanitizer variant matrix.

Each variant maps to one of the real MMM sanitizer entry points with a
documented set of parameters.  The parameter choices for the stealth
flags mirror the repo README's documented configurations:

* ``preserving``    — turbo preserving sanitizer, default flags
* ``paranoid``      — preserving sanitizer, paranoid mode
* ``max``           — paranoid + all advanced stealth flags (README
                      "Maximum stealth" command line)
* ``stealth_plus``  — the ``stealth-plus`` preset's advanced flags
                      (mmm/config/defaults.py)
* ``fast``          — fast_sanitize
* ``aggressive``    — aggressive_sanitize (effective sanitizer)
"""

from __future__ import annotations

from pathlib import Path


def _run_preserving(input_file, output_file, seed, kwargs):
    from mmm.preserving_sanitizer import preserving_sanitize

    params = dict(kwargs)
    params.setdefault("seed", seed)
    params.setdefault("threat_count", 0)
    return preserving_sanitize(input_file, output_file=output_file, **params)


def _run_fast(input_file, output_file, seed, kwargs):
    from mmm.fast_sanitizer import fast_sanitize

    params = dict(kwargs)
    params.setdefault("threat_count", 0)
    return fast_sanitize(input_file, output_file=output_file, **params)


def _run_aggressive(input_file, output_file, seed, kwargs):
    from mmm.effective_sanitizer import aggressive_sanitize

    params = dict(kwargs)
    params.setdefault("threat_count", 0)
    return aggressive_sanitize(input_file, output_file=output_file, **params)


VARIANTS = {
    "preserving": {
        "desc": "preserving sanitizer, default flags",
        "runner": _run_preserving,
        "kwargs": {},
    },
    "paranoid": {
        "desc": "preserving sanitizer, paranoid mode",
        "runner": _run_preserving,
        "kwargs": {"paranoid_mode": True},
    },
    "max": {
        "desc": "preserving sanitizer, paranoid + all advanced stealth flags (README maximum stealth)",
        "runner": _run_preserving,
        "kwargs": {
            "paranoid_mode": True,
            "masked_hf_phase": True,
            "gated_resample_nudge": True,
            "micro_eq_flutter": True,
            "hf_decorrelate": True,
            "adaptive_transient": True,
        },
    },
    "stealth_plus": {
        "desc": "preserving sanitizer, stealth-plus preset advanced flags",
        "runner": _run_preserving,
        "kwargs": {
            "paranoid_mode": True,
            "phase_dither": False,
            "comb_mask": False,
            "transient_shift": False,
            "phase_swirl": False,
            "masked_hf_phase": False,
            "resample_nudge": False,
            "gated_resample_nudge": True,
            "phase_noise": True,
            "micro_eq_flutter": False,
            "hf_decorrelate": False,
            "refined_transient": False,
            "adaptive_transient": False,
        },
    },
    "fast": {
        "desc": "fast sanitizer",
        "runner": _run_fast,
        "kwargs": {"paranoid_mode": False},
    },
    "aggressive": {
        "desc": "effective/aggressive sanitizer, paranoid mode",
        "runner": _run_aggressive,
        "kwargs": {"paranoid_mode": True},
    },
}

DEFAULT_VARIANTS = ["preserving", "paranoid"]


def resolve_variants(names: list[str] | str | None) -> list[tuple[str, dict]]:
    """Resolve a CLI variant list to (name, config) pairs."""
    if names is None:
        names = DEFAULT_VARIANTS
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",") if n.strip()]
    resolved = []
    for name in names:
        if name == "all":
            resolved.extend((n, c) for n, c in VARIANTS.items())
        elif name in VARIANTS:
            resolved.append((name, VARIANTS[name]))
        else:
            raise ValueError(f"unknown variant '{name}' (available: {', '.join(VARIANTS)})")
    return resolved


def variant_output_path(variant: str, input_file: Path, out_dir: Path) -> Path:
    """Output path for a processed file under the benchmark output dir."""
    out = out_dir / "audio" / variant
    out.mkdir(parents=True, exist_ok=True)
    return out / (input_file.stem + input_file.suffix)
