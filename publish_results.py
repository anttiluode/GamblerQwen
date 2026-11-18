"""Create self-contained data.js for the offline GitHub Pages dashboard.

The repository's JSON runs are the source of truth, not fabricated charts.
Use after running new experiments or uploading additional result JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_RUNS = (
    Path("results/reversal_alternate17.json"),
    Path("results/latest.json"),
)


def label(report: dict, name: str) -> str:
    meta = report["meta"]
    policy = meta.get("execution_policy", "unknown")
    scenario = meta.get("scenario", "?")
    if meta.get("source") == "qwen":
        tag = "Qwen proposes · forced alternating" if policy == "alternate" else "Qwen chooses and plays"
    else:
        tag = "Synthetic demo"
    return f"{tag} · {scenario} · {name}"


def bundle(paths: list[Path]) -> dict:
    runs = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; check in the real results or omit this run.")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data.get("rounds"), list) or not data["rounds"]:
            raise ValueError(f"{path}: missing recorded rounds")
        if not isinstance(data.get("meta"), dict):
            raise ValueError(f"{path}: missing metadata")
        rounds = data["rounds"]
        if any(r["round"] != i + 1 for i, r in enumerate(rounds)):
            raise ValueError(f"{path}: round numbering is not sequential")
        runs.append({"id": path.stem, "label": label(data, path.name), "data": data})
    return {"runs": runs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path,
                        help="JSON files (default: new alternating control, then original Qwen run)")
    parser.add_argument("--out", type=Path, default=Path("data.js"))
    args = parser.parse_args()
    selected = args.files or list(DEFAULT_RUNS)
    package = bundle(selected)
    encoded = json.dumps(package["runs"], separators=(",", ":"), ensure_ascii=True,
                         allow_nan=False).replace("<", "\\u003c")
    content = f"window.GAMBLER_RUNS={encoded};\nwindow.GAMBLER_DATA=window.GAMBLER_RUNS[0].data;\n"
    args.out.write_text(content, encoding="utf-8")
    print(f"Wrote {args.out} with {len(package['runs'])} genuine recordings:")
    for item in package["runs"]:
        print("  " + item["label"])
    print("Commit data.js to update the static website; no server needed.")


if __name__ == "__main__":
    main()
