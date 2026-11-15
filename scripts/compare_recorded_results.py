"""Compare re-run experiment JSON with the recorded results, value by value.

    python scripts/compare_recorded_results.py RERUN_RESULTS_JSON_DIR RECORDED_RESULTS_JSON_DIR
        [--since EPOCH_SECONDS] [--rtol 1e-9] [--output report.json]

Only files modified after ``--since`` (default: the ``.copied_at`` marker next to the re-run
``results`` directory, written by ``scripts/reproduce_manuscript.py``) are compared, so a file
that was merely copied and never regenerated is not counted as reproduced. Timing,
timestamp, environment and provenance fields are excluded; every other number, string and
boolean is compared (numbers with ``rtol``/``atol``; NaN equals NaN). Keys or list lengths
present in only one file are counted per side (schema drift between script versions). Exit code 0 if every compared
value agrees.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

VOLATILE = (
    "time",
    "seconds",
    "runtime",
    "duration",
    "elapsed",
    "timestamp",
    "date",
    "environment",
    "metadata",
    "version",
    "platform",
    "hostname",
    "git",
    "commit",
    "path",
    "python",
    "cpu",
    "memory",
    "rss",
    "sha256",
    "origin",
    "median_s",
    "iqr",
    "speed",
    "throughput",
    "overhead",
)


def volatile(key: str) -> bool:
    k = key.lower()
    return any(word in k for word in VOLATILE)


def walk(a, b, where, rtol, atol, stats, diffs):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if volatile(str(key)):
                continue
            if key not in a or key not in b:
                side = "re-run" if key in a else "recorded"
                stats["only_in_" + ("rerun" if key in a else "recorded")] += 1
                diffs.append(f"{where}/{key}: only in the {side} file")
                continue
            walk(a[key], b[key], f"{where}/{key}", rtol, atol, stats, diffs)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            stats["length_mismatch"] += 1
            diffs.append(f"{where}: length {len(a)} vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            walk(x, y, f"{where}[{i}]", rtol, atol, stats, diffs)
    elif isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or a is None:
        stats["compared"] += 1
        if a == b:
            stats["identical"] += 1
        else:
            stats["different"] += 1
            diffs.append(f"{where}: {a!r} vs {b!r}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        stats["compared"] += 1
        if a == b or (math.isnan(a) and math.isnan(b)):
            stats["identical"] += 1
        elif math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
            stats["close"] += 1
        else:
            stats["different"] += 1
            diffs.append(f"{where}: {a!r} vs {b!r}")
    else:
        stats["compared"] += 1
        stats["different"] += 1
        diffs.append(f"{where}: type {type(a).__name__} vs {type(b).__name__}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rerun", type=Path)
    parser.add_argument("recorded", type=Path)
    parser.add_argument("--since", type=float)
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument(
        "--atol", type=float, default=0.0, help="absolute tolerance (e.g. 1e-12 for error metrics)"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    since = args.since
    marker = args.rerun.parent.parent / ".copied_at"
    if since is None and marker.exists():
        since = float(marker.read_text().strip())
    report = {"rtol": args.rtol, "atol": args.atol, "since": since, "files": {}}
    failed = False
    for path in sorted(args.rerun.glob("*.json")):
        if path.name.endswith(".metadata.json") or "summary" in path.name:
            continue
        if since is not None and path.stat().st_mtime <= since:
            continue
        recorded = args.recorded / path.name
        if not recorded.exists():
            report["files"][path.name] = {"status": "no recorded counterpart"}
            continue
        stats = dict(
            compared=0,
            identical=0,
            close=0,
            different=0,
            only_in_rerun=0,
            only_in_recorded=0,
            length_mismatch=0,
        )
        diffs: list[str] = []
        walk(
            json.loads(path.read_text(encoding="utf-8")),
            json.loads(recorded.read_text(encoding="utf-8")),
            "",
            args.rtol,
            args.atol,
            stats,
            diffs,
        )
        # Values present in both files must agree; schema drift (keys or list lengths that
        # exist in only one file) is reported separately and also fails the comparison.
        ok = not (stats["different"] or stats["only_in_rerun"] or stats["only_in_recorded"])
        ok = ok and not stats["length_mismatch"]
        failed |= not ok
        report["files"][path.name] = {"status": "agrees" if ok else "differs", **stats}
        if diffs:
            report["files"][path.name]["first_differences"] = diffs[:10]
        print(f"{'OK  ' if ok else 'DIFF'} {path.name}: {stats}", flush=True)
    regenerated = len(report["files"])
    agree = sum(f.get("status") == "agrees" for f in report["files"].values())
    print(f"{agree}/{regenerated} regenerated result files agree with the recorded values")
    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
