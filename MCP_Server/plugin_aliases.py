"""Plugin parameter alias registry for known VST/AU plugins.

Maps friendly parameter names to real parameter names exposed by
the Live API. Used by MCP tools to resolve user-friendly names
before sending commands to the Remote Script.
"""

from typing import Optional, Dict

# Registry of known plugins with friendly parameter aliases and categories.
# Each plugin entry has:
#   "aliases": {friendly_name: real_param_name}
#   "categories": {category_name: [param_name_prefixes]}
KNOWN_PLUGINS: Dict[str, dict] = {
    "Serum": {
        "aliases": {
            "wavetable position": "Osc A WT Pos",
            "wavetable position a": "Osc A WT Pos",
            "wavetable position b": "Osc B WT Pos",
            "osc a level": "Osc A Level",
            "osc b level": "Osc B Level",
            "osc a pan": "Osc A Pan",
            "osc b pan": "Osc B Pan",
            "osc a unison": "Osc A Unison",
            "osc b unison": "Osc B Unison",
            "osc a detune": "Osc A UniDet",
            "osc b detune": "Osc B UniDet",
            "osc a octave": "Osc A Oct",
            "osc b octave": "Osc B Oct",
            "osc a semi": "Osc A Semi",
            "osc b semi": "Osc B Semi",
            "osc a fine": "Osc A Fine",
            "osc b fine": "Osc B Fine",
            "filter cutoff": "Fil Cutoff",
            "filter resonance": "Fil Reso",
            "filter drive": "Fil Drive",
            "filter mix": "Fil Mix",
            "filter pan": "Fil Pan",
            "filter fat": "Fil Fat",
            "noise level": "Noise Level",
            "noise pan": "Noise Pan",
            "sub level": "Sub Level",
            "sub pan": "Sub Pan",
            "macro 1": "Macro 1",
            "macro 2": "Macro 2",
            "macro 3": "Macro 3",
            "macro 4": "Macro 4",
            "macro 5": "Macro 5",
            "macro 6": "Macro 6",
            "macro 7": "Macro 7",
            "macro 8": "Macro 8",
            "master volume": "Master Vol",
        },
        "categories": {
            "Oscillator A": ["Osc A"],
            "Oscillator B": ["Osc B"],
            "Filter": ["Fil "],
            "Noise": ["Noise"],
            "Sub": ["Sub "],
            "Envelope": ["Env "],
            "LFO": ["LFO"],
            "Macro": ["Macro"],
            "Effects": ["FX"],
            "Master": ["Master"],
        },
    },
    # ── Native Ableton devices used for kick-drum sound design ──
    "Operator": {
        "aliases": {
            "osc a freq": "Osc A Frequency",
            "osc a tune": "Osc A Tune",
            "osc a level": "Osc A Level",
            "osc b freq": "Osc B Frequency",
            "osc b ratio": "Osc B Ratio",
            "osc b level": "Osc B Level",
            "fm amount": "Osc B->Osc A",
            "filter cutoff": "Filter Frequency",
            "filter resonance": "Filter Resonance",
            "pitch env decay": "Env A Decay",
            "pitch env amount": "Env A->Pitch",
            "amp env decay": "Env C Decay",
            "amp env attack": "Env C Attack",
        },
        "categories": {
            "Oscillator A": ["Osc A"],
            "Oscillator B": ["Osc B"],
            "Filter": ["Filter"],
            "Envelope": ["Env"],
            "LFO": ["LFO"],
            "Modulation": ["->"],
        },
    },
    "Saturator": {
        "aliases": {
            "drive": "Drive",
            "depth": "Depth",
            "dc": "DC",
            "balance": "Balance",
            "output": "Output",
            "mix": "Mix",
            "nonlinear": "Nonlinear",
        },
        "categories": {
            "Drive": ["Drive"],
            "Tone": ["DC", "Balance", "Offset"],
            "Output": ["Output", "Mix"],
        },
    },
    "EQ Eight": {
        "aliases": {
            "global gain": "Global Gain",
            "band 1 freq": "1 Frequency A",
            "band 1 gain": "1 Gain A",
            "band 1 q": "1 Resonance A",
            "band 1 type": "1 Filter Type A",
            "band 2 freq": "2 Frequency A",
            "band 2 gain": "2 Gain A",
            "band 2 q": "2 Resonance A",
            "band 2 type": "2 Filter Type A",
            "band 3 freq": "3 Frequency A",
            "band 3 gain": "3 Gain A",
            "band 3 q": "3 Resonance A",
            "band 3 type": "3 Filter Type A",
            "band 4 freq": "4 Frequency A",
            "band 4 gain": "4 Gain A",
            "band 4 q": "4 Resonance A",
            "band 4 type": "4 Filter Type A",
            "band 5 freq": "5 Frequency A",
            "band 5 gain": "5 Gain A",
            "band 5 q": "5 Resonance A",
            "band 5 type": "5 Filter Type A",
            "band 6 freq": "6 Frequency A",
            "band 6 gain": "6 Gain A",
            "band 6 q": "6 Resonance A",
            "band 6 type": "6 Filter Type A",
            "band 7 freq": "7 Frequency A",
            "band 7 gain": "7 Gain A",
            "band 7 q": "7 Resonance A",
            "band 7 type": "7 Filter Type A",
            "band 8 freq": "8 Frequency A",
            "band 8 gain": "8 Gain A",
            "band 8 q": "8 Resonance A",
            "band 8 type": "8 Filter Type A",
        },
        "categories": {
            "Band 1": ["1 "],
            "Band 2": ["2 "],
            "Band 3": ["3 "],
            "Band 4": ["4 "],
            "Band 5": ["5 "],
            "Band 6": ["6 "],
            "Band 7": ["7 "],
            "Band 8": ["8 "],
            "Global": ["Global"],
        },
    },
    "Pro-Q 3": {
        "aliases": {
            "band 1 freq": "Band 1 Frequency",
            "band 1 gain": "Band 1 Gain",
            "band 1 q": "Band 1 Q",
            "band 1 type": "Band 1 Shape",
            "band 2 freq": "Band 2 Frequency",
            "band 2 gain": "Band 2 Gain",
            "band 2 q": "Band 2 Q",
            "band 2 type": "Band 2 Shape",
            "band 3 freq": "Band 3 Frequency",
            "band 3 gain": "Band 3 Gain",
            "band 3 q": "Band 3 Q",
            "band 3 type": "Band 3 Shape",
            "band 4 freq": "Band 4 Frequency",
            "band 4 gain": "Band 4 Gain",
            "band 4 q": "Band 4 Q",
            "band 4 type": "Band 4 Shape",
            "output gain": "Output Gain",
        },
        "categories": {
            "Band 1": ["Band 1"],
            "Band 2": ["Band 2"],
            "Band 3": ["Band 3"],
            "Band 4": ["Band 4"],
            "Output": ["Output"],
        },
    },
    "Glue Compressor": {
        "aliases": {
            "threshold": "Threshold",
            "ratio": "Ratio",
            "attack": "Attack",
            "release": "Release",
            "makeup": "Makeup",
            "range": "Range",
            "sidechain": "SCC",
        },
        "categories": {
            "Compressor": ["Threshold", "Ratio", "Attack", "Release", "Makeup", "Range"],
            "Sidechain": ["SCC"],
        },
    },
    "Drum Buss": {
        "aliases": {
            "transient": "Transient",
            "boom": "Boom",
            "comp": "Compressor",
            "trim": "Trim",
            "dry wet": "Dry/Wet",
        },
        "categories": {
            "Drum": ["Transient", "Boom", "Compressor", "Trim"],
            "Mix": ["Dry/Wet"],
        },
    },
    "Corpus": {
        "aliases": {
            "resonator freq": "Frequency",
            "decay": "Decay",
            "resonance": "Resonance",
            "inharmonicity": "Inharmonicity",
            "output": "Output",
        },
        "categories": {
            "Resonator": ["Frequency", "Decay", "Resonance", "Inharmonicity"],
            "Output": ["Output"],
        },
    },
}


def resolve_alias(device_name: str, friendly_name: str) -> Optional[str]:
    """Resolve a friendly parameter name to the real parameter name.

    Returns the real parameter name if found, or None if no alias exists.
    Matching is case-insensitive on the friendly name.
    """
    profile = _find_profile(device_name)
    if profile is None:
        return None
    aliases = profile.get("aliases", {})
    friendly_lower = friendly_name.lower()
    for alias, real_name in aliases.items():
        if alias.lower() == friendly_lower:
            return real_name
    return None


def get_categories(device_name: str) -> Optional[Dict[str, list]]:
    """Get category definitions for a known plugin.

    Returns a dict of {category_name: [param_name_prefixes]},
    or None if the plugin is not in the registry.
    """
    profile = _find_profile(device_name)
    if profile is None:
        return None
    return profile.get("categories")


def get_alias_for_param(device_name: str, param_name: str) -> Optional[str]:
    """Get the friendly alias for a real parameter name.

    Returns the friendly alias if found, or None.
    """
    profile = _find_profile(device_name)
    if profile is None:
        return None
    aliases = profile.get("aliases", {})
    for alias, real_name in aliases.items():
        if real_name == param_name:
            return alias
    return None


def _find_profile(device_name: str) -> Optional[dict]:
    """Find a plugin profile by device name (case-insensitive contains)."""
    name_lower = device_name.lower()
    for plugin_name, profile in KNOWN_PLUGINS.items():
        if plugin_name.lower() in name_lower:
            return profile
    return None
