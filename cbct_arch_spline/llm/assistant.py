"""
Local-LLM assistant for the annotator GUI (via Ollama).

Turns a plain-English instruction ("make the panoramic brighter and extend the
arch back", "give me 30 control points") into a small, validated list of
operations the GUI knows how to apply. Nothing here executes arbitrary code:
the model may only choose from a fixed operation vocabulary, and every value is
range-checked before the GUI acts on it.

Runs fully locally against an Ollama server (default http://localhost:11434).
No API key, no network egress. Requires the user to have Ollama installed and a
model pulled (see check_ollama / DEFAULT_MODEL).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Optional

OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"
REQUEST_TIMEOUT = 120  # seconds; local generation can take a little while


# ---------------------------------------------------------------------------
# Operation vocabulary — the ONLY things the model is allowed to request.
# ---------------------------------------------------------------------------

# Panoramic render knobs → kwargs of synthesize_panoramic_from_volume_manual.
# Each entry: (min, max, human description) for prompt + validation.
PANO_PARAMS: dict[str, tuple[float, float, str]] = {
    "gamma":           (0.4, 2.0, "image gamma; <1 brighter, >1 darker (default 1.0)"),
    "strength":        (1.0, 8.0, "tone / overall contrast strength (default 4)"),
    "trough_depth_mm": (8.0, 24.0, "focal-trough depth in mm (default 14)"),
    "sup_margin_mm":   (20.0, 50.0, "extent above the arch, mm (default 38)"),
    "inf_margin_mm":   (8.0, 30.0, "extent below the arch, mm (default 16)"),
}
PANO_BOOL_PARAMS: dict[str, str] = {
    "clahe": "local adaptive contrast on/off",
}

ACTIONS = [
    "set_pano_param",       # param + value → change a panoramic render knob
    "regenerate_panoramic", # re-render the panoramic with current params
    "spline_resample",      # n → resample the arch to n evenly-spaced points
    "spline_smooth",        # re-fit the arch with more smoothing
    "spline_reorder",       # re-order control points along the arch
    "spline_clear",         # remove all control points
    "detect_geometric",     # run the geometric (ROI) auto arch detection
    "detect_ai",            # run the AI (HeatmapNet) detection
]

# JSON schema Ollama constrains the model's output to.
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "operations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ACTIONS},
                    "param": {"type": "string"},   # for set_pano_param
                    "value": {"type": "string"},   # value as string; coerced below
                    "n": {"type": "integer"},      # for spline_resample
                },
                "required": ["action"],
            },
        },
        "reply": {"type": "string"},
    },
    "required": ["operations", "reply"],
}


def _system_prompt(state: dict) -> str:
    pano_lines = "\n".join(
        f"    - {k}: {desc} (allowed {lo}–{hi})"
        for k, (lo, hi, desc) in PANO_PARAMS.items()
    )
    bool_lines = "\n".join(
        f"    - {k}: {desc} (value 'true' or 'false')"
        for k, desc in PANO_BOOL_PARAMS.items()
    )
    return f"""You control a dental CBCT annotation GUI. Convert the user's request into a
JSON list of operations chosen ONLY from the allowed actions. Do not invent actions.

Actions:
  - set_pano_param: change a panoramic render setting. Provide "param" and "value".
    Panoramic numeric settings:
{pano_lines}
    Panoramic on/off settings:
{bool_lines}
    To brighten, LOWER gamma (e.g. 0.7). To darken, raise gamma. More contrast:
    turn clahe true and/or raise strength.
  - regenerate_panoramic: re-render the panoramic (add this after changing pano settings).
  - spline_resample: set the number of arch control points. Provide integer "n" (8–40).
  - spline_smooth: make the arch smoother.
  - spline_reorder: re-order the arch points.
  - spline_clear: delete all arch points.
  - detect_geometric: auto-detect the arch from the image (no AI).
  - detect_ai: run the AI arch detector.

Current state:
  - control points on the arch: {state.get('n_points', 0)}
  - a panoramic exists: {state.get('has_pano', False)}
  - current panoramic settings: {json.dumps(state.get('pano_params', {}))}

Return JSON with "operations" (the list) and "reply" (one short sentence for the user).
Only include operations that are needed. Values must be within the allowed ranges."""


def _http_json(url: str, payload: Optional[dict] = None, timeout: float = 5.0) -> dict:
    """Minimal JSON GET/POST using the stdlib (no third-party deps)."""
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_ollama(model: str = DEFAULT_MODEL, host: str = OLLAMA_HOST) -> tuple[bool, str]:
    """
    Check that the Ollama server is reachable and the model is available.

    Returns (ok, message). message explains the fix when not ok.
    """
    try:
        data = _http_json(f"{host}/api/tags", timeout=5)
    except Exception:
        return False, (
            "Ollama server not reachable at "
            f"{host}.\nInstall it (`brew install ollama`), then run `ollama serve`."
        )
    names = [m.get("name", "") for m in data.get("models", [])]
    if not any(n == model or n.startswith(model + ":") or n.split(":")[0] == model.split(":")[0]
               for n in names):
        return False, (
            f"Model '{model}' is not pulled.\nRun:  ollama pull {model}"
        )
    return True, "ok"


def interpret(
    instruction: str,
    state: dict,
    model: str = DEFAULT_MODEL,
    host: str = OLLAMA_HOST,
) -> dict:
    """
    Ask the local model to turn `instruction` into validated operations.

    Returns {"operations": [...], "reply": str}. Raises RuntimeError on
    connection/parse failure (the GUI surfaces the message).
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _system_prompt(state)},
            {"role": "user", "content": instruction},
        ],
        "format": RESPONSE_SCHEMA,
        "stream": False,
        "options": {"temperature": 0.0},
    }
    try:
        data = _http_json(f"{host}/api/chat", payload=payload, timeout=REQUEST_TIMEOUT)
    except Exception as e:
        raise RuntimeError(f"Ollama request failed: {e}")

    content = data.get("message", {}).get("content", "")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        raise RuntimeError(f"Model did not return valid JSON:\n{content[:400]}")

    return {
        "operations": validate_operations(parsed.get("operations", [])),
        "reply": str(parsed.get("reply", "")).strip(),
    }


def validate_operations(ops: list) -> list[dict]:
    """Keep only well-formed, in-range operations; drop anything unrecognised."""
    clean: list[dict] = []
    for op in ops:
        if not isinstance(op, dict):
            continue
        action = op.get("action")
        if action not in ACTIONS:
            continue

        if action == "set_pano_param":
            param = op.get("param")
            value = _coerce(op.get("value"))
            if param in PANO_PARAMS and isinstance(value, (int, float)):
                lo, hi, _ = PANO_PARAMS[param]
                clean.append({"action": action, "param": param,
                              "value": float(min(max(value, lo), hi))})
            elif param in PANO_BOOL_PARAMS and isinstance(value, bool):
                clean.append({"action": action, "param": param, "value": value})
        elif action == "spline_resample":
            n = op.get("n")
            if isinstance(n, int):
                clean.append({"action": action, "n": int(min(max(n, 8), 40))})
        else:  # zero-argument actions
            clean.append({"action": action})
    return clean


def _coerce(value: Any) -> Any:
    """Coerce a string value to bool/float when possible."""
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "yes", "on"):
            return True
        if v in ("false", "no", "off"):
            return False
        try:
            return float(v)
        except ValueError:
            return value
    return value
