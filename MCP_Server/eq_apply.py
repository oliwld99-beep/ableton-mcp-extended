"""Map spectral EQ suggestions onto real EQ-device parameters.

Pure, device-agnostic helpers that turn the analyzer's EQSuggestions into
a concrete list of normalized (0..1) parameter writes for a specific EQ
device (Ableton EQ Eight, FabFilter Pro-Q, and similar band-based EQs).

These functions never talk to Ableton; they take the parameter metadata
returned by ``get_device_parameters`` (name/min/max/is_quantized/
value_items) and produce a plan. The MCP tool executes the plan via
``set_device_parameter`` (which expects normalized values).

Kept separate from the analyzer so the mapping/normalization logic is
unit-testable with mocked parameter lists.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# filter_type (analyzer vocabulary) -> substrings to look for in a device's
# quantized filter-type value_items (case-insensitive).
FILTER_TYPE_KEYWORDS: Dict[str, List[str]] = {
    "bell": ["bell", "peak"],
    "high_pass": ["low cut", "high pass", "hi pass", "high-pass", "hpf"],
    "low_pass": ["high cut", "low pass", "lo pass", "low-pass", "lpf"],
    "high_shelf": ["high shelf", "hi shelf", "high-shelf"],
    "low_shelf": ["low shelf", "lo shelf", "low-shelf"],
}


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def normalized(value: float, pmin: float, pmax: float) -> float:
    """Linear normalize a real value into 0..1 using a param's min/max."""
    if pmax == pmin:
        return 0.0
    return clamp01((float(value) - float(pmin)) / (float(pmax) - float(pmin)))


def match_filter_index(filter_type: str, value_items: List[str]) -> Optional[int]:
    """Find the value_items index matching a desired filter type, or None."""
    if not value_items:
        return None
    keywords = FILTER_TYPE_KEYWORDS.get(filter_type, [filter_type.replace("_", " ")])
    for kw in keywords:
        for idx, item in enumerate(value_items):
            if kw in str(item).lower():
                return idx
    return None


def _classify_param(name: str) -> Tuple[Optional[int], Optional[str]]:
    """Return (band_number, role) for an EQ parameter name.

    role in {freq, gain, q, type, on}. Returns (None, None) when the
    parameter is not a recognizable band control.
    """
    low = name.lower()
    m = re.search(r"\d+", name)
    band = int(m.group()) if m else None

    if band is None:
        return None, None

    if "filter on" in low or "enabled" in low or low.rstrip().endswith(" on a") or low.rstrip().endswith(" on"):
        role = "on"
    elif "freq" in low:
        role = "freq"
    elif "gain" in low:
        role = "gain"
    elif "resonance" in low or low.rstrip().endswith(" q") or low.rstrip().endswith(" q a") or " q " in low:
        role = "q"
    elif "shape" in low or "type" in low:
        role = "type"
    else:
        role = None
    return band, role


def index_of_bands(parameters: List[dict]) -> Dict[int, Dict[str, dict]]:
    """Group a device's parameters into {band_number: {role: param}}.

    Parameters are the dicts from get_device_parameters. When several
    params map to the same (band, role) the first one wins (this keeps the
    primary "A" curve on EQ Eight rather than the alternate "B" curve).
    """
    bands: Dict[int, Dict[str, dict]] = {}
    for p in parameters:
        band, role = _classify_param(p.get("name", ""))
        if band is None or role is None:
            continue
        # Skip alternate "B" curve on EQ Eight (dual mode).
        if re.search(r"\bB\b", p.get("name", "")):
            continue
        slot = bands.setdefault(band, {})
        slot.setdefault(role, p)
    return bands


def plan_eq_application(
    device_name: str,
    parameters: List[dict],
    suggestions: List[dict],
    max_bands: int = 8,
) -> Tuple[List[dict], List[str]]:
    """Build a normalized write plan for applying suggestions to an EQ.

    ``suggestions`` are dicts with keys: frequency, gain_db, q,
    filter_type. Returns ``(plan, warnings)`` where plan entries are:
        {band, param_name, role, normalized, target}
    ``target`` is a human-readable description of the intended value.
    """
    bands = index_of_bands(parameters)
    warnings: List[str] = []
    if not bands:
        warnings.append(
            f"Device '{device_name}' has no recognizable EQ bands; "
            "cannot map suggestions automatically."
        )
        return [], warnings

    available = sorted(b for b, roles in bands.items() if "freq" in roles)
    plan: List[dict] = []

    for i, sug in enumerate(suggestions):
        if i >= len(available) or i >= max_bands:
            warnings.append(
                f"Ran out of EQ bands: {len(suggestions)} suggestions but only "
                f"{min(len(available), max_bands)} usable band(s)."
            )
            break

        band_no = available[i]
        roles = bands[band_no]
        ftype = sug.get("filter_type", "bell")

        # Enable the band if there is an on/enabled toggle.
        if "on" in roles:
            plan.append({
                "band": band_no,
                "param_name": roles["on"]["name"],
                "role": "on",
                "normalized": 1.0,
                "target": "enabled",
            })

        # Filter type (quantized).
        if "type" in roles:
            tp = roles["type"]
            idx = match_filter_index(ftype, tp.get("value_items", []))
            items = tp.get("value_items", [])
            if idx is not None and len(items) > 1:
                plan.append({
                    "band": band_no,
                    "param_name": tp["name"],
                    "role": "type",
                    "normalized": round(idx / (len(items) - 1), 4),
                    "target": f"{ftype} ({items[idx]})",
                })
            else:
                warnings.append(
                    f"Band {band_no}: filter type '{ftype}' not available on "
                    f"'{device_name}'; leaving type unchanged."
                )

        # Frequency.
        if "freq" in roles:
            fp = roles["freq"]
            plan.append({
                "band": band_no,
                "param_name": fp["name"],
                "role": "freq",
                "normalized": round(normalized(sug["frequency"], fp["min"], fp["max"]), 4),
                "target": f"{sug['frequency']} Hz",
            })

        # Gain (skip for pass filters).
        if ftype not in ("high_pass", "low_pass") and "gain" in roles:
            gp = roles["gain"]
            plan.append({
                "band": band_no,
                "param_name": gp["name"],
                "role": "gain",
                "normalized": round(normalized(sug.get("gain_db", 0.0), gp["min"], gp["max"]), 4),
                "target": f"{sug.get('gain_db', 0.0):+} dB",
            })

        # Q (bell only, best-effort).
        if ftype == "bell" and "q" in roles and sug.get("q"):
            qp = roles["q"]
            plan.append({
                "band": band_no,
                "param_name": qp["name"],
                "role": "q",
                "normalized": round(normalized(sug["q"], qp["min"], qp["max"]), 4),
                "target": f"Q~{sug['q']} (approx)",
            })

    return plan, warnings
