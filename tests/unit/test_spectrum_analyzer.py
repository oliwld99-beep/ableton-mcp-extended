"""Unit tests for the offline spectrum analyzer (MCP_Server.spectrum_analyzer)."""
import os
import tempfile
import wave

import numpy as np
import pytest

from MCP_Server import spectrum_analyzer as sa


# ── helpers ─────────────────────────────────────────────────────────
def _write_wav(samples: np.ndarray, sr: int = 44100, channels: int = 1) -> str:
    """Write a float [-1,1] signal to a temp 16-bit WAV and return its path."""
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767.0).astype(np.int16)
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


def _sine(freq: float, sr: int = 44100, dur: float = 1.0, amp: float = 0.8) -> np.ndarray:
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return amp * np.sin(2 * np.pi * freq * t)


# ── audio loading ───────────────────────────────────────────────────
class TestLoadAudio:
    def test_load_mono_wav(self):
        path = _write_wav(_sine(440), channels=1)
        try:
            samples, sr, ch = sa.load_audio(path)
            assert sr == 44100
            assert ch == 1
            assert samples.ndim == 1
            assert np.max(np.abs(samples)) > 0.5
        finally:
            os.remove(path)

    def test_load_stereo_downmix(self):
        sig = _sine(440)
        stereo = np.repeat(sig[:, None], 2, axis=1).reshape(-1)
        path = _write_wav(stereo, channels=2)
        try:
            samples, sr, ch = sa.load_audio(path)
            assert ch == 2
            assert samples.ndim == 1
        finally:
            os.remove(path)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            sa.load_audio("/nonexistent/path/to/file.wav")


# ── spectrum estimation ─────────────────────────────────────────────
class TestAverageSpectrum:
    def test_peak_at_sine_frequency(self):
        sr = 44100
        freqs, mag = sa.average_spectrum(_sine(1000, sr=sr, dur=1.0), sr, fft_size=4096)
        peak_freq = freqs[int(np.argmax(mag))]
        assert abs(peak_freq - 1000) < 20  # within a bin or two

    def test_short_signal_is_zero_padded(self):
        sr = 44100
        short = _sine(500, sr=sr, dur=0.01)  # fewer than fft_size samples
        freqs, mag = sa.average_spectrum(short, sr, fft_size=4096)
        assert mag.size == 4096 // 2 + 1

    def test_empty_signal_raises(self):
        with pytest.raises(ValueError):
            sa.average_spectrum(np.array([]), 44100)


# ── resonance detection ─────────────────────────────────────────────
class TestResonanceDetection:
    def test_detects_injected_resonance(self):
        sr = 44100
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(sr) * 0.05
        sig = noise + _sine(250, sr=sr, dur=1.0, amp=0.5)
        freqs, mag = sa.average_spectrum(sig, sr)
        res = sa.detect_resonances(freqs, mag)
        assert res, "expected at least one resonance"
        assert abs(res[0].freq - 250) < 15
        assert res[0].excess_db > 4.0

    def test_flat_noise_has_few_resonances(self):
        sr = 44100
        rng = np.random.default_rng(1)
        noise = rng.standard_normal(sr) * 0.2
        freqs, mag = sa.average_spectrum(noise, sr)
        res = sa.detect_resonances(freqs, mag, threshold_db=6.0)
        assert len(res) <= 3

    def test_q_within_bounds(self):
        sr = 44100
        sig = _sine(500, sr=sr) + np.random.default_rng(2).standard_normal(sr) * 0.05
        freqs, mag = sa.average_spectrum(sig, sr)
        for r in sa.detect_resonances(freqs, mag):
            assert 0.5 <= r.q <= 12.0


# ── global metrics ──────────────────────────────────────────────────
class TestGlobalMetrics:
    def test_bright_signal_has_higher_centroid(self):
        sr = 44100
        lo = _sine(200, sr=sr)
        hi = _sine(8000, sr=sr)
        f1, m1 = sa.average_spectrum(lo, sr)
        f2, m2 = sa.average_spectrum(hi, sr)
        c_lo = sa.global_metrics(lo, sr, f1, m1)["spectral_centroid_hz"]
        c_hi = sa.global_metrics(hi, sr, f2, m2)["spectral_centroid_hz"]
        assert c_hi > c_lo

    def test_region_energy_sums_to_roughly_100(self):
        sr = 44100
        sig = _sine(440, sr=sr) + _sine(3000, sr=sr) * 0.5
        f, m = sa.average_spectrum(sig, sr)
        metrics = sa.global_metrics(sig, sr, f, m)
        total = sum(metrics["region_energy_pct"].values())
        assert 90.0 <= total <= 100.5

    def test_crest_factor_positive(self):
        sr = 44100
        sig = _sine(440, sr=sr)
        f, m = sa.average_spectrum(sig, sr)
        metrics = sa.global_metrics(sig, sr, f, m)
        assert metrics["crest_factor_db"] > 0


# ── EQ suggestions ──────────────────────────────────────────────────
class TestEQSuggestions:
    def test_resonance_produces_bell_cut(self):
        res = [sa.Resonance(freq=250.0, excess_db=10.0, q=4.0)]
        metrics = {"region_energy_pct": {"sub": 50.0}, "spectral_tilt_db_per_oct": -3.0}
        out = sa.suggest_eq(metrics, res, "techno", "bass")
        bells = [s for s in out if s.filter_type == "bell"]
        assert bells
        assert bells[0].gain_db < 0
        assert abs(bells[0].frequency - 250.0) < 1

    def test_highpass_suggested_for_hats(self):
        metrics = {"region_energy_pct": {"sub": 2.0}, "spectral_tilt_db_per_oct": -3.0}
        out = sa.suggest_eq(metrics, [], "techno", "hats")
        assert any(s.filter_type == "high_pass" for s in out)

    def test_no_highpass_when_sub_is_intended_content(self):
        # A sub-heavy signal (sub region dominant) should not get an HPF.
        metrics = {"region_energy_pct": {"sub": 60.0}, "spectral_tilt_db_per_oct": -3.0}
        out = sa.suggest_eq(metrics, [], "techno", "sub")
        assert not any(s.filter_type == "high_pass" for s in out)

    def test_bright_spectrum_gets_high_shelf_cut(self):
        metrics = {"region_energy_pct": {"sub": 10.0}, "spectral_tilt_db_per_oct": 2.0}
        out = sa.suggest_eq(metrics, [], "lo-fi", "synth")
        shelves = [s for s in out if s.filter_type == "high_shelf"]
        assert shelves
        assert shelves[0].gain_db < 0

    def test_unknown_genre_and_instrument_fall_back(self):
        metrics = {"region_energy_pct": {"sub": 10.0}, "spectral_tilt_db_per_oct": -3.0}
        # Should not raise on unknown keys.
        out = sa.suggest_eq(metrics, [], "polka", "kazoo")
        assert isinstance(out, list)


# ── octave bands ────────────────────────────────────────────────────
class TestOctaveBands:
    def test_band_count_and_type(self):
        sr = 44100
        f, m = sa.average_spectrum(_sine(440, sr=sr), sr)
        bands = sa.octave_bands(f, m)
        assert len(bands) > 10
        assert all(isinstance(b, sa.Band) for b in bands)


# ── end-to-end ──────────────────────────────────────────────────────
class TestAnalyzeEndToEnd:
    def test_full_pipeline_and_report(self):
        sr = 44100
        rng = np.random.default_rng(3)
        sig = rng.standard_normal(sr) * 0.05 + _sine(120, sr=sr, amp=0.6)
        path = _write_wav(sig)
        try:
            result = sa.analyze(path, genre="techno", instrument="bass")
            assert result.sample_rate == 44100
            assert result.global_metrics
            assert result.bands
            report = sa.format_report(result)
            assert "Spectral Analysis" in report
            assert "EQ suggestions" in report or "balanced" in report
            # round-trips to dict for the json format
            d = result.to_dict()
            assert d["genre"] == "techno"
        finally:
            os.remove(path)

    def test_max_seconds_truncation(self):
        sr = 8000
        sig = _sine(300, sr=sr, dur=5.0)
        path = _write_wav(sig, sr=sr)
        try:
            result = sa.analyze(path, max_seconds=1.0)
            # duration reflects full file, analysis still succeeds
            assert result.duration_s >= 4.0
            assert result.global_metrics
        finally:
            os.remove(path)
