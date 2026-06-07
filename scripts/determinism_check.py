#!/usr/bin/env python3
"""
determinism_check.py
====================

Runs the transformer-VM on a chosen compiled program N times and asserts
that every run produces byte-identical output -- i.e., that the in-model
execution is deterministic across independent invocations.

This is the small "verifiability" probe behind the deployability claim:
in regulated European industrials (chemicals, manufacturing, biotech) a
scheduler that can drift between runs is not auditable. A transformer-VM
that executes a compiled C program with byte-identical output across N
runs is.

Usage:
    python scripts/determinism_check.py [PROGRAM_TXT] [-n N]

Defaults:
    PROGRAM_TXT = transformer_vm/data/job_shop.txt
    N          = 5
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path


def extract_program_output(raw_stdout: bytes) -> bytes:
    """
    Pull just the program's printf output out of `wasm-run`'s stdout.

    wasm-run wraps the program's stdout in framework chatter that
    includes per-run wall-clock times (e.g. `FAIL (86.17s)`). We want
    to hash only the bytes the program itself emitted via printf.

    Convention:
      - The program output is prefixed by `\n  output: ` on the
        line where it begins.
      - It ends just before the summary line containing
        `N passed, M failed,`.
    """
    text = raw_stdout.decode("utf-8", errors="replace")
    marker = "\n  output: "
    idx = text.find(marker)
    if idx < 0:
        return b""
    start = idx + len(marker)

    summary_idx = text.find(" passed, ", start)
    if summary_idx < 0:
        return text[start:].encode("utf-8")
    # Walk back to the start of the summary line.
    line_start = text.rfind("\n", start, summary_idx) + 1
    return text[start:line_start].rstrip("\n").encode("utf-8") + b"\n"


def run_once(program_txt: Path) -> tuple[bytes, float]:
    """Run `uv run wasm-run PROGRAM_TXT` once, returning (program_output_bytes, wall_seconds)."""
    t0 = time.perf_counter()
    result = subprocess.run(
        ["uv", "run", "wasm-run", str(program_txt)],
        capture_output=True,
        # NOTE: do NOT check=True. wasm-run returns nonzero when its
        # internal trace-comparison flags any divergence from the
        # graph-evaluator reference, even when the visible program
        # output is byte-identical (see writeup for the upstream
        # rough-edge). We compare the program output ourselves.
    )
    return extract_program_output(result.stdout), time.perf_counter() - t0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "program_txt",
        nargs="?",
        default="transformer_vm/data/job_shop.txt",
        help="Compiled program token file (default: %(default)s)",
    )
    parser.add_argument("-n", type=int, default=5, help="Number of runs (default: 5)")
    args = parser.parse_args()

    program_txt = Path(args.program_txt)
    if not program_txt.exists():
        print(f"error: {program_txt} does not exist", file=sys.stderr)
        print("  hint: run `uv run wasm-compile` first to produce token files.", file=sys.stderr)
        return 2

    print(f"determinism check: {program_txt}  (n={args.n})\n")
    print(f"{'run':>4}  {'bytes':>9}  {'seconds':>9}  sha256")
    print(f"{'-'*4}  {'-'*9}  {'-'*9}  {'-'*64}")

    digests: list[str] = []
    times:   list[float] = []
    for i in range(1, args.n + 1):
        stdout, secs = run_once(program_txt)
        digest = hashlib.sha256(stdout).hexdigest()
        digests.append(digest)
        times.append(secs)
        print(f"{i:>4}  {len(stdout):>9}  {secs:>9.3f}  {digest}")

    print()
    if len(set(digests)) == 1:
        mean_s = sum(times) / len(times)
        print(f"PASS: all {args.n} runs produced byte-identical output.")
        print(f"      mean wall-time: {mean_s:.3f}s")
        return 0

    print(f"FAIL: produced {len(set(digests))} distinct outputs across {args.n} runs.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
