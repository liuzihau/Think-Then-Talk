"""Compare two `capture_inference.py` JSON outputs.

Reports prompt-segment, generated-segment, and decoded-text mismatches with
the first divergence index. Exit code is 0 on full match, 1 on any mismatch
(so CI / shell scripts can gate on `$?`).

Usage:

    python tests/equivalence/diff_tokens.py /tmp/baseline.json /tmp/refactor.json

    # Allow up to 2 token mismatches in the generated segment (for prefer_no_mask
    # kernel-routing tolerance):
    python tests/equivalence/diff_tokens.py baseline.json engine.json --max_diff 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _first_diff_index(a: list, b: list) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return -1 if len(a) == len(b) else n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("baseline", type=Path)
    p.add_argument("current", type=Path)
    p.add_argument("--max_diff", type=int, default=0,
                   help="Allow up to N token mismatches in generated segment "
                        "(0 = byte-identical required).")
    args = p.parse_args()

    a = _load(args.baseline)
    b = _load(args.current)

    print(f"baseline: mode={a['mode']}  commit={a.get('git_commit','?')[:10]}  nfe={a.get('nfe','?')}")
    print(f"current : mode={b['mode']}  commit={b.get('git_commit','?')[:10]}  nfe={b.get('nfe','?')}")

    failures: list[str] = []

    # 1. Prompt tokens.
    a_p, b_p = a["prompt_token_ids"], b["prompt_token_ids"]
    if a_p != b_p:
        idx = _first_diff_index(a_p, b_p)
        failures.append(
            f"prompt_token_ids differ at index {idx}: "
            f"baseline len={len(a_p)} current len={len(b_p)} — "
            "tokenizer or chat template diverged?"
        )

    # 2. Generated tokens.
    a_g, b_g = a["generated_token_ids"], b["generated_token_ids"]
    if len(a_g) != len(b_g):
        failures.append(
            f"generated_token_ids length mismatch: baseline={len(a_g)} current={len(b_g)}"
        )
    else:
        diffs = [i for i, (x, y) in enumerate(zip(a_g, b_g)) if x != y]
        if diffs:
            head = diffs[:8]
            print(f"generated tokens differ at {len(diffs)} / {len(a_g)} positions; "
                  f"first 8: {head}")
            for i in head:
                print(f"  [{i:3d}] baseline={a_g[i]:6d}  current={b_g[i]:6d}")
            if len(diffs) > args.max_diff:
                failures.append(
                    f"{len(diffs)} generated-token mismatches exceed --max_diff={args.max_diff}"
                )
            else:
                print(f"WITHIN tolerance (max_diff={args.max_diff})")

    # 3. Decoded text — informational (not a hard gate; tokens are authoritative).
    if a["decoded_text"] != b["decoded_text"]:
        print("decoded_text differs:")
        print(f"  baseline: {a['decoded_text'][:160]}")
        print(f"  current : {b['decoded_text'][:160]}")
    else:
        print("decoded_text matches")

    if failures:
        print("\nFAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
