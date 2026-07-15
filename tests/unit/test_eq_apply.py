"""Unit tests for EQ suggestion -> device parameter mapping (MCP_Server.eq_apply)."""
import pytest

from MCP_Server import eq_apply

# EQ Eight-like filter type items (order matters for index tests).
EQ8_TYPES = [
    "Low Cut 48", "Low Cut 12", "Low Shelf", "Bell",
    "Notch", "High Shelf", "High Cut 12", "High Cut 48",
]
PROQ_SHAPES = [
    "Bell", "Low Shelf", "Low Cut", "High Shelf", "High Cut",
    "Notch", "Band Pass", "Tilt Shelf",
]


def _eq8_band(n):
    """Return a set of EQ Eight-like parameter dicts for band n."""
    return [
        {"name": f"{n} Filter On A", "min": 0, "max": 1, "is_quantized": True,
         "value_items": ["Off", "On"]},
        {"name": f"{n} Frequency A", "min": 10.0, "max": 22000.0,
         "is_quantized": False, "value_items": []},
        {"name": f"{n} Gain A", "min": -15.0, "max": 15.0,
         "is_quantized": False, "value_items": []},
        {"name": f"{n} Resonance A", "min": 0.1, "max": 10.0,
         "is_quantized": False, "value_items": []},
        {"name": f"{n} Filter Type A", "min": 0, "max": len(EQ8_TYPES) - 1,
         "is_quantized": True, "value_items": EQ8_TYPES},
    ]


class TestNormalized:
    def test_linear_mid(self):
        assert eq_apply.normalized(0, -15, 15) == pytest.approx(0.5)

    def test_clamped(self):
        assert eq_apply.normalized(100, -15, 15) == 1.0
        assert eq_apply.normalized(-100, -15, 15) == 0.0

    def test_zero_range(self):
        assert eq_apply.normalized(5, 3, 3) == 0.0


class TestFilterIndex:
    def test_bell_eq8(self):
        assert eq_apply.match_filter_index("bell", EQ8_TYPES) == 3

    def test_high_pass_maps_to_low_cut(self):
        assert eq_apply.match_filter_index("high_pass", EQ8_TYPES) == 0

    def test_high_shelf(self):
        assert eq_apply.match_filter_index("high_shelf", EQ8_TYPES) == 5

    def test_proq_bell(self):
        assert eq_apply.match_filter_index("bell", PROQ_SHAPES) == 0

    def test_proq_high_pass_maps_to_low_cut(self):
        assert eq_apply.match_filter_index("high_pass", PROQ_SHAPES) == 2

    def test_unknown_returns_none(self):
        assert eq_apply.match_filter_index("bell", []) is None


class TestBandIndexing:
    def test_groups_bands_and_roles(self):
        params = _eq8_band(1) + _eq8_band(2)
        bands = eq_apply.index_of_bands(params)
        assert set(bands.keys()) == {1, 2}
        assert set(bands[1].keys()) == {"on", "freq", "gain", "q", "type"}

    def test_skips_alternate_b_curve(self):
        params = _eq8_band(1) + [
            {"name": "1 Frequency B", "min": 10, "max": 22000,
             "is_quantized": False, "value_items": []},
        ]
        bands = eq_apply.index_of_bands(params)
        # The A curve should remain the mapped frequency param.
        assert bands[1]["freq"]["name"] == "1 Frequency A"


class TestPlanApplication:
    def test_bell_plan_values(self):
        params = _eq8_band(1)
        suggestions = [{"frequency": 250.0, "gain_db": -6.0, "q": 4.0, "filter_type": "bell"}]
        plan, warnings = eq_apply.plan_eq_application("EQ Eight", params, suggestions)
        roles = {e["role"]: e for e in plan}
        assert roles["on"]["normalized"] == 1.0
        assert roles["type"]["target"].startswith("bell")
        assert roles["freq"]["normalized"] == pytest.approx((250 - 10) / (22000 - 10), abs=1e-3)
        assert roles["gain"]["normalized"] == pytest.approx((-6 + 15) / 30, abs=1e-3)
        assert "q" in roles
        assert not warnings

    def test_highpass_has_no_gain(self):
        params = _eq8_band(1)
        suggestions = [{"frequency": 30.0, "gain_db": 0.0, "q": 0.7, "filter_type": "high_pass"}]
        plan, _ = eq_apply.plan_eq_application("EQ Eight", params, suggestions)
        roles = {e["role"] for e in plan}
        assert "gain" not in roles
        assert "freq" in roles

    def test_runs_out_of_bands_warns(self):
        params = _eq8_band(1)  # only one band
        suggestions = [
            {"frequency": 100.0, "gain_db": -2.0, "q": 2.0, "filter_type": "bell"},
            {"frequency": 500.0, "gain_db": -3.0, "q": 2.0, "filter_type": "bell"},
        ]
        plan, warnings = eq_apply.plan_eq_application("EQ Eight", params, suggestions)
        assert any("Ran out of EQ bands" in w for w in warnings)
        # Only band 1 used.
        assert all(e["band"] == 1 for e in plan)

    def test_no_bands_returns_warning(self):
        params = [{"name": "Dry/Wet", "min": 0, "max": 1, "is_quantized": False, "value_items": []}]
        suggestions = [{"frequency": 100.0, "gain_db": -2.0, "q": 2.0, "filter_type": "bell"}]
        plan, warnings = eq_apply.plan_eq_application("Reverb", params, suggestions)
        assert plan == []
        assert warnings
