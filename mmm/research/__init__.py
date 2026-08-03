"""
mmm.research — Offline robustness assessment harness.

Implements the *offline, sandboxed* measurement phase of engagement
SUB-AHA-2026-SEC-001 (Robustness Assessment of Audio Identification and
AI-Generated Content Detection).  Everything in this package runs locally
on synthetic / open-licensed audio; it does NOT call any live production
service (SubmitHub AI Song Checker, AHA Music, ACRCloud, ...).

Design goals
------------
* Reuse the real MMM sanitization code paths (preserving / fast /
  aggressive sanitizers) instead of reimplementing them.
* Measure the quantities named in the Scope of Work:
    - ΔMatch       : reduction in acoustic-fingerprint match confidence
                     (surrogate: spectral-peak hashing, Shazam-style)
    - ΔAI score    : reduction in AI-generated-content probability
                     (surrogate: interpretable feature-direction proxy,
                     NOT the SubmitHub model)
    - objective quality: SNR, log-spectral distance, spectral angle,
                     STOI (speech-oriented, optional)
* Be transparent about what is a *surrogate* and what is authoritative.
  The sponsors' sandboxed API endpoints / evaluation libraries are the
  authoritative instruments; until they are provided, the surrogates
  here are directional research aids only.
"""

__version__ = "0.1.0"
