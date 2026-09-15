#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def lower_is_better(a, b):
    if a is None or b is None:
        return "unavailable"
    if a == b:
        return "tie"
    return "surya" if a < b else "chandra"


def higher_is_better(a, b):
    if a is None or b is None:
        return "unavailable"
    if a == b:
        return "tie"
    return "surya" if a > b else "chandra"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--surya", required=True)
    p.add_argument("--chandra", required=True)
    p.add_argument("--out", default="artifacts/evaluation/comparison.json")
    args = p.parse_args()

    surya = load(args.surya)
    chandra = load(args.chandra)

    sm = surya["aggregate"]
    cm = chandra["aggregate"]

    result = {
        "surya": surya,
        "chandra": chandra,
        "winner_by_metric": {
            "weighted_CER": lower_is_better(
                sm.get("weighted_CER"),
                cm.get("weighted_CER"),
            ),
            "weighted_WER": lower_is_better(
                sm.get("weighted_WER"),
                cm.get("weighted_WER"),
            ),
            "reading_order_inversion_rate": lower_is_better(
                sm.get("mean_reading_order_inversion_rate"),
                cm.get("mean_reading_order_inversion_rate"),
            ),
            "structure_macro_f1": higher_is_better(
                sm.get("mean_structure_macro_f1"),
                cm.get("mean_structure_macro_f1"),
            ),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
