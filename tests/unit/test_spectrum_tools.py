"""Unit tests for the spectral-analysis MCP tools in MCP_Server.server."""
import os
import sys
import tempfile
import wave
from unittest.mock import MagicMock, patch

import numpy as np

# Mock MCP dependencies before importing the server module (same pattern as
# the other server tool tests).
_mock_fastmcp = MagicMock()
_mock_fastmcp.FastMCP.return_value.tool.return_value = lambda fn: fn
sys.modules.setdefault("mcp", MagicMock())
sys.modules.setdefault("mcp.server", MagicMock())
sys.modules["mcp.server.fastmcp"] = _mock_fastmcp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from MCP_Server.server import (  # noqa: E402
    analyze_audio_file,
    analyze_clip_spectrum,
    apply_eq_suggestions,
)

_EQ8_TYPES = [
    "Low Cut 48", "Low Cut 12", "Low Shelf", "Bell",
    "Notch", "High Shelf", "High Cut 12", "High Cut 48",
]


def _eq8_params():
    params = []
    for n in (1, 2, 3):
        params += [
            {"name": f"{n} Filter On A", "min": 0, "max": 1,
             "is_quantized": True, "value_items": ["Off", "On"]},
            {"name": f"{n} Frequency A", "min": 10.0, "max": 22000.0,
             "is_quantized": False, "value_items": []},
            {"name": f"{n} Gain A", "min": -15.0, "max": 15.0,
             "is_quantized": False, "value_items": []},
            {"name": f"{n} Resonance A", "min": 0.1, "max": 10.0,
             "is_quantized": False, "value_items": []},
            {"name": f"{n} Filter Type A", "min": 0, "max": len(_EQ8_TYPES) - 1,
             "is_quantized": True, "value_items": _EQ8_TYPES},
        ]
    return {"device_name": "EQ Eight", "parameters": params}


def _make_wav(freq=250.0, sr=44100, dur=1.0):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(t.size) * 0.05 + 0.5 * np.sin(2 * np.pi * freq * t)
    sig = np.clip(sig, -1, 1)
    pcm = (sig * 32767).astype(np.int16)
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


class TestAnalyzeAudioFile:
    def test_report_contains_expected_sections(self):
        path = _make_wav()
        try:
            out = analyze_audio_file(MagicMock(), path, genre="techno", instrument="bass")
            assert "Spectral Analysis" in out
            assert "Levels:" in out
            assert "Hz" in out
        finally:
            os.remove(path)

    def test_json_format(self):
        path = _make_wav()
        try:
            out = analyze_audio_file(MagicMock(), path, format="json")
            import json

            data = json.loads(out)
            assert data["sample_rate"] == 44100
            assert "eq_suggestions" in data
        finally:
            os.remove(path)

    def test_missing_file_returns_error_string(self):
        out = analyze_audio_file(MagicMock(), "/no/such/file.wav")
        assert out.startswith("Error")


class TestAnalyzeClipSpectrum:
    @patch("MCP_Server.server.get_ableton_connection")
    def test_analyzes_audio_clip(self, mock_conn):
        path = _make_wav()
        mock_ableton = MagicMock()
        mock_ableton.send_command.return_value = {
            "track_index": 0,
            "clip_index": 0,
            "name": "MyLoop",
            "is_audio": True,
            "file_path": path,
        }
        mock_conn.return_value = mock_ableton
        try:
            out = analyze_clip_spectrum(MagicMock(), 1, 1, genre="techno", instrument="bass")
            assert "MyLoop" in out
            assert "Spectral Analysis" in out
        finally:
            os.remove(path)

    @patch("MCP_Server.server.get_ableton_connection")
    def test_midi_clip_returns_error(self, mock_conn):
        mock_ableton = MagicMock()
        mock_ableton.send_command.return_value = {
            "error": "clip is not an audio clip",
            "is_audio": False,
        }
        mock_conn.return_value = mock_ableton
        out = analyze_clip_spectrum(MagicMock(), 1, 1)
        assert "Cannot analyze clip" in out

    @patch("MCP_Server.server.get_ableton_connection")
    def test_no_file_path_returns_helpful_message(self, mock_conn):
        mock_ableton = MagicMock()
        mock_ableton.send_command.return_value = {
            "name": "Recorded",
            "is_audio": True,
            "file_path": "",
        }
        mock_conn.return_value = mock_ableton
        out = analyze_clip_spectrum(MagicMock(), 1, 1)
        assert "no" in out.lower() and "sample file path" in out

    @patch("MCP_Server.server.get_ableton_connection")
    def test_track_index_converted_to_zero_based(self, mock_conn):
        path = _make_wav()
        mock_ableton = MagicMock()
        mock_ableton.send_command.return_value = {
            "name": "L", "is_audio": True, "file_path": path,
        }
        mock_conn.return_value = mock_ableton
        try:
            analyze_clip_spectrum(MagicMock(), 3, 2)
            args, kwargs = mock_ableton.send_command.call_args
            assert args[0] == "get_clip_sample_path"
            assert args[1]["track_index"] == 2  # 3 -> 2 (0-based)
            assert args[1]["clip_index"] == 1   # 2 -> 1 (0-based)
        finally:
            os.remove(path)


_SUGGESTIONS = (
    '[{"frequency": 250.0, "gain_db": -6.0, "q": 4.0, "filter_type": "bell"},'
    ' {"frequency": 30.0, "gain_db": 0.0, "q": 0.71, "filter_type": "high_pass"}]'
)


class TestApplyEQSuggestions:
    @patch("MCP_Server.server.get_ableton_connection")
    def test_dry_run_previews_without_writing(self, mock_conn):
        mock_ableton = MagicMock()

        def side_effect(command, params=None):
            if command == "get_device_parameters":
                return _eq8_params()
            raise AssertionError(f"unexpected write in dry_run: {command}")

        mock_ableton.send_command.side_effect = side_effect
        mock_conn.return_value = mock_ableton

        out = apply_eq_suggestions(
            MagicMock(), 1, 1, suggestions_json=_SUGGESTIONS, dry_run=True
        )
        assert "DRY-RUN" in out
        assert "band 1" in out
        # No set_device_parameter was called.
        calls = [c.args[0] for c in mock_ableton.send_command.call_args_list]
        assert "set_device_parameter" not in calls

    @patch("MCP_Server.server.get_ableton_connection")
    def test_apply_writes_parameters(self, mock_conn):
        mock_ableton = MagicMock()

        def side_effect(command, params=None):
            if command == "get_device_parameters":
                return _eq8_params()
            if command == "set_device_parameter":
                return {"parameter_name": params.get("parameter_name"), "new_value": 0}
            raise AssertionError(f"unexpected: {command}")

        mock_ableton.send_command.side_effect = side_effect
        mock_conn.return_value = mock_ableton

        out = apply_eq_suggestions(
            MagicMock(), 1, 1, suggestions_json=_SUGGESTIONS, dry_run=False
        )
        assert "Applied" in out
        writes = [c for c in mock_ableton.send_command.call_args_list
                  if c.args[0] == "set_device_parameter"]
        assert len(writes) >= 4  # freq/gain/type/on across bands

    @patch("MCP_Server.server.get_ableton_connection")
    def test_bad_json_returns_error(self, mock_conn):
        mock_conn.return_value = MagicMock()
        out = apply_eq_suggestions(MagicMock(), 1, 1, suggestions_json="{not json")
        assert out.startswith("Error")

    def test_requires_source(self):
        out = apply_eq_suggestions(MagicMock(), 1, 1)
        assert "provide either" in out
