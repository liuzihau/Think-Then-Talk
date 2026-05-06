"""Typed reveal/decode policy overrides for the eval harness.

The eval CLI surface (`eval_t3.sh` → JSON `model_args` → `eval_t3.py`'s
harness `__init__`) ships a flat list of optional knobs that override the
checkpoint's training-time denoise schedule. This module groups them into a
single dataclass plus the small typed parsers used to coerce values out of
strings/None.

Construct via `DenoiseOverrides.from_kwargs(**kw)` (kwargs come straight from
the lm_eval CLI as strings) and then call `.apply(model_config)` to fold the
overrides into `model_config["denoise"]`. When `use_sh_denoise_policy` is
False, `apply()` is a no-op — the checkpoint's policy wins.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Iterable, Optional

from utils import get_denoise_decode_config, get_denoise_reveal_config


# ---- Typed coercers ----------------------------------------------------------

def is_missing(value: Any) -> bool:
    return value is None or value == ""


def as_optional_int(value: Any) -> Optional[int]:
    return None if is_missing(value) else int(value)


def as_optional_float(value: Any) -> Optional[float]:
    return None if is_missing(value) else float(value)


def as_optional_str(value: Any) -> Optional[str]:
    return None if is_missing(value) else str(value)


def as_optional_bool(value: Any) -> Optional[bool]:
    if is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean override from value: {value!r}")


# ---- Dataclass ---------------------------------------------------------------

_REVEAL_WEIGHT_FIELDS = ("reveal_random", "reveal_greedy", "reveal_ar_force")
_DECODE_WEIGHT_FIELDS = ("decode_fix", "decode_greedy")


@dataclass
class DenoiseOverrides:
    """Reveal/decode policy overrides consumed by the eval harness."""

    use_sh_denoise_policy:        Optional[bool]  = None

    reveal_k:                     Optional[int]   = None
    reveal_policy_mode:           Optional[str]   = None
    reveal_random_weight:         Optional[float] = None
    reveal_greedy_weight:         Optional[float] = None
    reveal_ar_force_weight:       Optional[float] = None

    decode_policy_mode:           Optional[str]   = None
    decode_fix_k:                 Optional[int]   = None
    decode_max_k:                 Optional[int]   = None
    decode_min_k:                 Optional[int]   = None
    decode_confidence_threshold:  Optional[float] = None
    decode_fix_weight:            Optional[float] = None
    decode_greedy_weight:         Optional[float] = None

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "DenoiseOverrides":
        """Build from a flat kwargs dict (e.g. lm_eval's --model_args JSON).

        Unknown keys are ignored; missing keys default to None. String values
        are coerced to the field's declared optional type.
        """
        coercers = {
            "use_sh_denoise_policy":        as_optional_bool,
            "reveal_k":                     as_optional_int,
            "reveal_policy_mode":           as_optional_str,
            "reveal_random_weight":         as_optional_float,
            "reveal_greedy_weight":         as_optional_float,
            "reveal_ar_force_weight":       as_optional_float,
            "decode_policy_mode":           as_optional_str,
            "decode_fix_k":                 as_optional_int,
            "decode_max_k":                 as_optional_int,
            "decode_min_k":                 as_optional_int,
            "decode_confidence_threshold":  as_optional_float,
            "decode_fix_weight":            as_optional_float,
            "decode_greedy_weight":         as_optional_float,
        }
        return cls(**{name: coercers[name](kwargs.get(name)) for name in coercers})

    # ---- Application ---------------------------------------------------------

    def _raw_overrides(self) -> Dict[str, Any]:
        """Per-field map (all fields except `use_sh_denoise_policy`) as raw values."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name != "use_sh_denoise_policy"
        }

    def _require_explicit(self) -> None:
        """When use_sh_denoise_policy is on, every other field must be set."""
        missing = [name for name, value in self._raw_overrides().items() if value is None]
        if missing:
            raise ValueError(
                "USE_SH_DENOISE_POLICY=1 requires every reveal/decode override to be set explicitly. "
                "Missing: " + ", ".join(sorted(missing))
            )

    def apply(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Fold these overrides into `model_config["denoise"]` and return it.

        - When `use_sh_denoise_policy` is unset/false: no-op (checkpoint wins).
        - Otherwise: every field must be set (raises if not), and the
          override values replace the checkpoint's denoise schedule.
        """
        if not bool(self.use_sh_denoise_policy):
            return model_config
        self._require_explicit()

        denoise_cfg = dict(model_config.get("denoise", {}) or {})
        reveal_cfg = get_denoise_reveal_config(denoise_cfg)
        decode_cfg = get_denoise_decode_config(denoise_cfg, default_k=reveal_cfg["k"])

        if self.reveal_k is not None:
            reveal_cfg["k"] = int(self.reveal_k)
        reveal_cfg["policy"] = _build_policy_override(
            reveal_cfg.get("policy"),
            fixed_mode=self.reveal_policy_mode,
            weight_overrides={
                "random":   self.reveal_random_weight,
                "greedy":   self.reveal_greedy_weight,
                "ar_force": self.reveal_ar_force_weight,
            },
        )

        if self.decode_fix_k is not None:
            decode_cfg["fix_k"] = int(self.decode_fix_k)
        if self.decode_max_k is not None:
            decode_cfg["max_k"] = int(self.decode_max_k)
        if self.decode_min_k is not None:
            decode_cfg["min_k"] = int(self.decode_min_k)
        if self.decode_confidence_threshold is not None:
            decode_cfg["confidence_threshold"] = float(self.decode_confidence_threshold)
        decode_cfg["policy"] = _build_policy_override(
            decode_cfg.get("policy"),
            fixed_mode=self.decode_policy_mode,
            weight_overrides={
                "fix":    self.decode_fix_weight,
                "greedy": self.decode_greedy_weight,
            },
        )

        denoise_cfg["reveal"] = reveal_cfg
        denoise_cfg["decode"] = decode_cfg
        model_config["denoise"] = denoise_cfg
        return model_config


# ---- Policy-override helpers ------------------------------------------------

def _extract_policy_choices(policy: Any, valid_modes: Iterable[str]) -> Dict[str, float]:
    """Pull mixture weights out of an existing policy spec, defaulting to 0.0."""
    choices = {mode: 0.0 for mode in valid_modes}
    if isinstance(policy, str) and policy in choices:
        choices[policy] = 1.0
        return choices
    if isinstance(policy, dict):
        raw = policy.get("choices", {})
        if isinstance(raw, dict):
            for mode in choices:
                if mode in raw:
                    choices[mode] = float(raw[mode])
    return choices


def _build_policy_override(
    current_policy: Any,
    *,
    fixed_mode: Optional[str],
    weight_overrides: Dict[str, Optional[float]],
) -> Any:
    """Return the new policy after applying overrides.

    A `fixed_mode` (single string) replaces the policy outright. Otherwise
    any non-None weights are merged into the existing mixture weights. If
    no weight overrides are given, the existing policy passes through.
    """
    if fixed_mode is not None:
        return fixed_mode
    provided = {mode: w for mode, w in weight_overrides.items() if w is not None}
    if not provided:
        return current_policy
    merged = _extract_policy_choices(current_policy, weight_overrides.keys())
    for mode, w in provided.items():
        merged[mode] = float(w)
    return {"type": "mixture", "choices": merged}
