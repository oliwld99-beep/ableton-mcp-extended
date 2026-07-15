"""Offline spectral analysis and EQ-suggestion engine.

This module performs real FFT-based analysis of audio *files* (samples,
loops, one-shots) and derives concrete, genre/instrument-aware EQ
suggestions.  It is the honest, working answer to "can Producer Pal
analyze audio?": it cannot hear the live master bus (that needs a
Max-for-Live FFT probe), but it *can* read the sample file behind a
clip and analyze it accurately.

Design goals:
    * numpy-only hard dependency (FFT + vector maths).
    * WAV decoding via the standard-library ``wave`` module, so plain
      WAV files work with zero extra dependencies.  Non-WAV formats
      (aif/flac/mp3/...) are decoded via the optional ``soundfile``
      package when it is installed.
    * Pure functions that are trivially unit-testable with synthetic
      signals.

Nothing here talks to Ableton; the MCP layer wires this to a clip's
sample path.
"""

from __future__ import annotations

import math
import os
import wave
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# ── Reference band centers (1/3-octave, 20 Hz .. 20 kHz) ───────────
THIRD_OCTAVE_CENTERS: List[float] = [
    20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400,
    500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000,
    6300, 8000, 10000, 12500, 16000, 20000,
]

# Broad tonal regions used for balance reporting.
REGIONS: List[Tuple[str, float, float]] = [
    ("sub", 20.0, 60.0),
    ("low", 60.0, 250.0),
    ("low_mid", 250.0, 800.0),
    ("mid", 800.0, 2500.0),
    ("high_mid", 2500.0, 6000.0),
    ("high", 6000.0, 12000.0),
    ("air", 12000.0, 20000.0),
]

EPS = 1e-12


# ── Data classes ───────────────────────────────────────────────────
@dataclass
class Band:
    freq: float
    db: float


@dataclass
class Resonance:
    freq: float
    excess_db: float          # how far the peak sticks out above the smoothed baseline
    q: float                  # estimated quality factor


@dataclass
class EQSuggestion:
    frequency: float
    gain_db: float
    q: float
    filter_type: str          # bell | high_pass | low_pass | high_shelf | low_shelf
    reason: str
    confidence: float         # 0..1


@dataclass
class AnalysisResult:
    file_path: str
    sample_rate: int
    duration_s: float
    channels: int
    genre: str
    instrument: str
    global_metrics: dict = field(default_factory=dict)
    bands: List[Band] = field(default_factory=list)
    resonances: List[Resonance] = field(default_factory=list)
    eq_suggestions: List[EQSuggestion] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ── Audio loading ──────────────────────────────────────────────────
def load_audio(path: str) -> Tuple[np.ndarray, int, int]:
    """Load an audio file to mono float32 in [-1, 1].

    Returns ``(samples_mono, sample_rate, channels)``.

    WAV is decoded with the stdlib.  Other formats require ``soundfile``.
    Raises ``FileNotFoundError`` / ``ValueError`` on problems.
    """
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        try:
            return _load_wav_stdlib(path)
        except Exception:
            # Fall through to soundfile (e.g. float/extensible WAV).
            pass

    try:
        import soundfile as sf  # type: ignore
    except ImportError as exc:
        raise ValueError(
            f"Cannot decode '{ext or 'unknown'}' files without the optional "
            "'soundfile' dependency. Install with: pip install soundfile"
        ) from exc

    data, sr = sf.read(path, always_2d=True, dtype="float32")
    channels = data.shape[1]
    mono = data.mean(axis=1).astype(np.float64)
    return mono, int(sr), int(channels)


def _load_wav_stdlib(path: str) -> Tuple[np.ndarray, int, int]:
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 1:  # unsigned 8-bit
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    elif sampwidth == 3:  # packed 24-bit
        a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        ints = (a[:, 0] | (a[:, 1] << 8) | (a[:, 2] << 16))
        ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
        data = ints.astype(np.float64) / 8388608.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if n_channels > 1:
        data = data.reshape(-1, n_channels)
        mono = data.mean(axis=1)
    else:
        mono = data
    return mono, int(sr), int(n_channels)


# ── Spectrum estimation ────────────────────────────────────────────
def average_spectrum(
    samples: np.ndarray,
    sample_rate: int,
    fft_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """Welch-style averaged magnitude spectrum.

    Returns ``(freqs, magnitude_linear)`` where magnitude is the average
    RMS-normalized magnitude per FFT bin.
    """
    if samples.size == 0:
        raise ValueError("Empty audio signal")

    fft_size = int(fft_size)
    if samples.size < fft_size:
        # Zero-pad short signals so we still get one frame.
        padded = np.zeros(fft_size, dtype=np.float64)
        padded[: samples.size] = samples
        samples = padded

    window = np.hanning(fft_size)
    win_power = np.sum(window ** 2)
    hop = fft_size // 2

    n_frames = 1 + (samples.size - fft_size) // hop
    acc = np.zeros(fft_size // 2 + 1, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop
        frame = samples[start : start + fft_size] * window
        spec = np.fft.rfft(frame)
        # Power spectral density normalized by window power.
        acc += (np.abs(spec) ** 2) / win_power
    acc /= max(n_frames, 1)

    mag = np.sqrt(acc)
    freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    return freqs, mag


def to_db(mag_linear: np.ndarray, ref: float = 1.0) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(mag_linear, EPS) / ref)


# ── Band aggregation ───────────────────────────────────────────────
def octave_bands(
    freqs: np.ndarray,
    mag_linear: np.ndarray,
    centers: Optional[List[float]] = None,
) -> List[Band]:
    """Aggregate FFT bins into 1/3-octave bands (power sum -> dB)."""
    centers = centers or THIRD_OCTAVE_CENTERS
    nyquist = freqs[-1] if freqs.size else 0.0
    factor = 2 ** (1.0 / 6.0)  # half a 1/3-octave, in ratio
    power = mag_linear ** 2
    ref = _reference_power(freqs, power)
    bands: List[Band] = []
    for fc in centers:
        if fc > nyquist:
            break
        lo, hi = fc / factor, fc * factor
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            # Nearest-bin fallback for sparse low-frequency bands.
            idx = int(np.argmin(np.abs(freqs - fc)))
            band_power = power[idx]
        else:
            band_power = np.sum(power[mask])
        db = 10.0 * math.log10(max(band_power, EPS) / ref)
        bands.append(Band(freq=float(fc), db=round(db, 2)))
    return bands


def _reference_power(freqs: np.ndarray, power: np.ndarray) -> float:
    """Reference = total broadband power, so band dB values are relative."""
    total = float(np.sum(power))
    return max(total, EPS)


# ── Resonance detection ────────────────────────────────────────────
def detect_resonances(
    freqs: np.ndarray,
    mag_linear: np.ndarray,
    threshold_db: float = 4.0,
    max_peaks: int = 8,
    f_min: float = 30.0,
    f_max: float = 18000.0,
) -> List[Resonance]:
    """Detect narrow resonances that stick out from a smoothed baseline.

    The signal is resampled onto a log-frequency grid, a ~1-octave moving
    average provides the perceptual baseline, and local maxima of the
    residual above ``threshold_db`` are reported as resonances.
    """
    if freqs.size < 4:
        return []

    f_max = min(f_max, float(freqs[-1]))
    if f_min >= f_max:
        return []

    # Log grid at 1/24-octave resolution.
    steps_per_octave = 24
    n_octaves = math.log2(f_max / f_min)
    n_points = max(int(n_octaves * steps_per_octave), 8)
    log_grid = f_min * (2 ** (np.linspace(0, n_octaves, n_points)))

    mag_db = to_db(mag_linear)
    grid_db = np.interp(log_grid, freqs, mag_db)

    # ~1 octave smoothing window (odd length).
    win = steps_per_octave | 1
    baseline = _moving_average(grid_db, win)
    residual = grid_db - baseline

    peaks: List[Resonance] = []
    for i in range(1, n_points - 1):
        if residual[i] < threshold_db:
            continue
        if residual[i] <= residual[i - 1] or residual[i] < residual[i + 1]:
            continue  # not a local maximum
        q = _estimate_q(log_grid, residual, i)
        peaks.append(
            Resonance(
                freq=round(float(log_grid[i]), 1),
                excess_db=round(float(residual[i]), 2),
                q=round(q, 2),
            )
        )

    peaks.sort(key=lambda r: r.excess_db, reverse=True)
    return peaks[:max_peaks]


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.copy()
    pad = win // 2
    padded = np.pad(x, pad, mode="edge")
    kernel = np.ones(win) / win
    return np.convolve(padded, kernel, mode="valid")


def _estimate_q(grid: np.ndarray, residual: np.ndarray, peak_idx: int) -> float:
    """Estimate Q from the -3 dB width of the residual peak on the log grid."""
    peak_val = residual[peak_idx]
    target = peak_val - 3.0
    lo = peak_idx
    while lo > 0 and residual[lo] > target:
        lo -= 1
    hi = peak_idx
    while hi < len(residual) - 1 and residual[hi] > target:
        hi += 1
    f_lo, f_hi = grid[lo], grid[hi]
    bandwidth = max(f_hi - f_lo, EPS)
    q = grid[peak_idx] / bandwidth
    # Clamp to musically sensible range.
    return float(min(max(q, 0.5), 12.0))


# ── Global metrics ─────────────────────────────────────────────────
def global_metrics(
    samples: np.ndarray,
    sample_rate: int,
    freqs: np.ndarray,
    mag_linear: np.ndarray,
) -> dict:
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
    peak_db = 20.0 * math.log10(max(peak, EPS))
    rms_db = 20.0 * math.log10(max(rms, EPS))
    crest_db = peak_db - rms_db

    power = mag_linear ** 2
    total_power = float(np.sum(power)) + EPS

    # Spectral centroid (brightness).
    centroid = float(np.sum(freqs * power) / total_power)

    # Spectral flatness (0 tonal .. 1 noise-like).
    valid = mag_linear > EPS
    if np.any(valid):
        geo = math.exp(float(np.mean(np.log(mag_linear[valid]))))
        arith = float(np.mean(mag_linear[valid]))
        flatness = geo / max(arith, EPS)
    else:
        flatness = 0.0

    # Spectral tilt: slope of dB vs log-freq (dB/octave), least squares.
    tilt = _spectral_tilt(freqs, mag_linear)

    region_energy = _region_energy(freqs, power, total_power)

    return {
        "peak_db": round(peak_db, 2),
        "rms_db": round(rms_db, 2),
        "crest_factor_db": round(crest_db, 2),
        "spectral_centroid_hz": round(centroid, 1),
        "spectral_flatness": round(flatness, 4),
        "spectral_tilt_db_per_oct": round(tilt, 2),
        "region_energy_pct": region_energy,
    }


def _spectral_tilt(freqs: np.ndarray, mag_linear: np.ndarray) -> float:
    mask = (freqs >= 40.0) & (freqs <= 16000.0) & (mag_linear > EPS)
    if np.count_nonzero(mask) < 4:
        return 0.0
    x = np.log2(freqs[mask])
    y = to_db(mag_linear[mask])
    slope = float(np.polyfit(x, y, 1)[0])
    return slope


def _region_energy(freqs: np.ndarray, power: np.ndarray, total: float) -> dict:
    out = {}
    for name, lo, hi in REGIONS:
        mask = (freqs >= lo) & (freqs < hi)
        out[name] = round(100.0 * float(np.sum(power[mask])) / total, 1)
    return out


# ── Genre / instrument profiles ────────────────────────────────────
# highpass: recommended HPF (Hz) for this instrument, 0 = none.
# fundamental_range: expected musical body (Hz), used to protect it from cuts.
INSTRUMENT_PROFILES = {
    "kick": {"highpass": 25, "fundamental_range": (40, 100), "boom": (150, 400)},
    "sub": {"highpass": 20, "fundamental_range": (30, 90), "boom": (120, 300)},
    "bass": {"highpass": 30, "fundamental_range": (40, 250), "boom": (250, 500)},
    "snare": {"highpass": 90, "fundamental_range": (150, 250), "boom": (300, 600)},
    "hats": {"highpass": 300, "fundamental_range": (0, 0), "boom": (200, 500)},
    "perc": {"highpass": 150, "fundamental_range": (0, 0), "boom": (200, 600)},
    "vocal": {"highpass": 80, "fundamental_range": (100, 350), "boom": (200, 500)},
    "synth": {"highpass": 40, "fundamental_range": (80, 500), "boom": (200, 500)},
    "pad": {"highpass": 60, "fundamental_range": (100, 500), "boom": (200, 500)},
    "lead": {"highpass": 100, "fundamental_range": (150, 800), "boom": (250, 600)},
    "pluck": {"highpass": 120, "fundamental_range": (150, 700), "boom": (250, 600)},
    "guitar": {"highpass": 80, "fundamental_range": (100, 400), "boom": (200, 500)},
    "keys": {"highpass": 40, "fundamental_range": (80, 500), "boom": (200, 500)},
    "fullmix": {"highpass": 0, "fundamental_range": (0, 0), "boom": (200, 500)},
    "master": {"highpass": 0, "fundamental_range": (0, 0), "boom": (200, 500)},
    "other": {"highpass": 0, "fundamental_range": (0, 0), "boom": (200, 500)},
}

# target_tilt: desired dB/octave slope (negative = darker/warmer).
GENRE_PROFILES = {
    "techno": {"target_tilt": -3.0, "air_boost": 1.0, "warmth": 0.0},
    "house": {"target_tilt": -3.0, "air_boost": 1.5, "warmth": 0.0},
    "hip-hop": {"target_tilt": -4.0, "air_boost": 1.0, "warmth": 1.0},
    "trap": {"target_tilt": -4.0, "air_boost": 1.5, "warmth": 0.5},
    "lo-fi": {"target_tilt": -6.0, "air_boost": -2.0, "warmth": 2.0},
    "dnb": {"target_tilt": -3.0, "air_boost": 2.0, "warmth": 0.0},
    "pop": {"target_tilt": -3.5, "air_boost": 2.0, "warmth": 0.5},
    "rock": {"target_tilt": -3.5, "air_boost": 1.5, "warmth": 0.5},
    "orchestral": {"target_tilt": -4.0, "air_boost": 1.0, "warmth": 1.0},
    "ambient": {"target_tilt": -5.0, "air_boost": 0.0, "warmth": 1.5},
    "default": {"target_tilt": -3.5, "air_boost": 1.0, "warmth": 0.5},
}


def _instrument_profile(instrument: str) -> dict:
    return INSTRUMENT_PROFILES.get(instrument.lower(), INSTRUMENT_PROFILES["other"])


def _genre_profile(genre: str) -> dict:
    return GENRE_PROFILES.get(genre.lower(), GENRE_PROFILES["default"])


# ── EQ suggestion engine ───────────────────────────────────────────
def suggest_eq(
    metrics: dict,
    resonances: List[Resonance],
    genre: str,
    instrument: str,
    max_suggestions: int = 6,
) -> List[EQSuggestion]:
    inst = _instrument_profile(instrument)
    gen = _genre_profile(genre)
    suggestions: List[EQSuggestion] = []

    # 1) High-pass to remove sub-rumble below the instrument's body.
    hpf = inst["highpass"]
    if hpf and metrics.get("region_energy_pct", {}).get("sub", 0.0) < 40.0:
        # Only suggest HPF when the sub region isn't the intended content.
        suggestions.append(
            EQSuggestion(
                frequency=float(hpf),
                gain_db=0.0,
                q=0.71,
                filter_type="high_pass",
                reason=f"Remove inaudible rumble below {hpf} Hz for a cleaner {instrument}.",
                confidence=0.7,
            )
        )

    # 2) Resonance cuts (protect the musical fundamental).
    fund_lo, fund_hi = inst["fundamental_range"]
    for res in resonances:
        in_fundamental = fund_lo <= res.freq <= fund_hi and (fund_hi - fund_lo) > 0
        cut = -min(res.excess_db * 0.75, 6.0)
        if in_fundamental:
            cut *= 0.5  # be gentle inside the body
        suggestions.append(
            EQSuggestion(
                frequency=res.freq,
                gain_db=round(cut, 1),
                q=res.q,
                filter_type="bell",
                reason=f"Resonance +{res.excess_db} dB detected at {res.freq} Hz.",
                confidence=round(min(0.5 + res.excess_db / 20.0, 0.95), 2),
            )
        )

    # 3) Tonal-balance shelves based on genre target tilt.
    tilt = metrics.get("spectral_tilt_db_per_oct", 0.0)
    target = gen["target_tilt"]
    tilt_err = tilt - target  # positive => too bright vs. genre target
    if abs(tilt_err) >= 1.5:
        if tilt_err > 0:
            suggestions.append(
                EQSuggestion(
                    frequency=8000.0,
                    gain_db=round(-min(tilt_err, 4.0), 1),
                    q=0.5,
                    filter_type="high_shelf",
                    reason=f"Spectrum is brighter ({tilt:+.1f} dB/oct) than typical {genre} "
                    f"({target:+.1f}); tame the top.",
                    confidence=0.55,
                )
            )
        else:
            suggestions.append(
                EQSuggestion(
                    frequency=10000.0,
                    gain_db=round(min(-tilt_err, 3.0), 1),
                    q=0.5,
                    filter_type="high_shelf",
                    reason=f"Spectrum is duller ({tilt:+.1f} dB/oct) than typical {genre} "
                    f"({target:+.1f}); add air.",
                    confidence=0.5,
                )
            )

    # 4) Genre "air" nudge for full mixes.
    if instrument.lower() in ("fullmix", "master") and gen["air_boost"] > 0:
        suggestions.append(
            EQSuggestion(
                frequency=12000.0,
                gain_db=round(gen["air_boost"], 1),
                q=0.5,
                filter_type="high_shelf",
                reason=f"Gentle air boost for {genre} sheen.",
                confidence=0.4,
            )
        )

    # De-duplicate near-identical frequencies, keep the highest |gain|.
    suggestions = _dedupe_suggestions(suggestions)
    suggestions.sort(key=lambda s: s.confidence, reverse=True)
    return suggestions[:max_suggestions]


def _dedupe_suggestions(items: List[EQSuggestion]) -> List[EQSuggestion]:
    kept: List[EQSuggestion] = []
    for item in items:
        dup = False
        for k in kept:
            if k.filter_type == item.filter_type and abs(
                math.log2(max(k.frequency, 1) / max(item.frequency, 1))
            ) < 0.17:  # within ~1/6 octave
                dup = True
                if abs(item.gain_db) > abs(k.gain_db):
                    k.gain_db = item.gain_db
                    k.reason = item.reason
                break
        if not dup:
            kept.append(item)
    return kept


# ── Top-level orchestration ────────────────────────────────────────
def analyze(
    path: str,
    genre: str = "default",
    instrument: str = "other",
    fft_size: int = 4096,
    max_seconds: float = 30.0,
) -> AnalysisResult:
    """Full analysis pipeline for one audio file."""
    samples, sr, channels = load_audio(path)
    duration = samples.size / float(sr) if sr else 0.0

    # Cap analysis length for very long files.
    if max_seconds and samples.size > int(max_seconds * sr):
        samples = samples[: int(max_seconds * sr)]

    freqs, mag = average_spectrum(samples, sr, fft_size=fft_size)
    metrics = global_metrics(samples, sr, freqs, mag)
    bands = octave_bands(freqs, mag)
    resonances = detect_resonances(freqs, mag)
    eq = suggest_eq(metrics, resonances, genre, instrument)

    return AnalysisResult(
        file_path=path,
        sample_rate=sr,
        duration_s=round(duration, 3),
        channels=channels,
        genre=genre,
        instrument=instrument,
        global_metrics=metrics,
        bands=bands,
        resonances=resonances,
        eq_suggestions=eq,
    )


def format_report(result: AnalysisResult) -> str:
    """Human-readable summary for the MCP tool response."""
    m = result.global_metrics
    lines = [
        f"=== Spectral Analysis: {os.path.basename(result.file_path)} ===",
        f"Genre: {result.genre} | Instrument: {result.instrument} | "
        f"{result.sample_rate} Hz | {result.channels}ch | {result.duration_s}s",
        "",
        "Levels:",
        f"  Peak: {m['peak_db']} dBFS | RMS: {m['rms_db']} dBFS | "
        f"Crest: {m['crest_factor_db']} dB",
        f"  Centroid: {m['spectral_centroid_hz']} Hz | "
        f"Flatness: {m['spectral_flatness']} | Tilt: {m['spectral_tilt_db_per_oct']} dB/oct",
        "",
        "Energy by region (%):",
    ]
    reg = m.get("region_energy_pct", {})
    lines.append("  " + " | ".join(f"{k}: {v}" for k, v in reg.items()))
    lines.append("")

    if result.resonances:
        lines.append("Resonances (excess above baseline):")
        for r in result.resonances:
            lines.append(f"  {r.freq} Hz  +{r.excess_db} dB  (Q~{r.q})")
    else:
        lines.append("Resonances: none significant.")
    lines.append("")

    if result.eq_suggestions:
        lines.append("EQ suggestions:")
        for s in result.eq_suggestions:
            gain = "" if s.filter_type in ("high_pass", "low_pass") else f"{s.gain_db:+} dB, "
            lines.append(
                f"  [{s.filter_type}] {s.frequency} Hz, {gain}Q={s.q}  "
                f"-> {s.reason} (conf {s.confidence})"
            )
    else:
        lines.append("EQ suggestions: spectrum looks balanced, no changes needed.")

    return "\n".join(lines)
