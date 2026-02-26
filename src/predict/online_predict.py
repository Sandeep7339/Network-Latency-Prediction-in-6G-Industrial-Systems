"""
Online prediction script – loads the stacking ensemble and preprocessing
pipeline, reads a JSON batch of flow records, and writes predictions.

Usage
-----
    python -m src.predict.online_predict             # default paths
    python -m src.predict.online_predict \\
        --input  examples/sample_input.json \\
        --output examples/sample_output.json

Input JSON format::

    {
      "flows": [
        { "timestamp_ns": ..., "src_device_id": ..., ... },
        ...
      ]
    }

Output JSON format::

    {
      "predictions": [
        {
          "flow_index": 0,
          "predicted_latency_us": 99.12,
          "violation_probability": 0.07,
          "violation_flag": false
        },
        ...
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# ── Make sure project root is importable ─────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# These imports register custom classes so joblib.load works
from src.features.feature_pipeline import FrequencyEncoder  # noqa: F401
from src.models.ensemble import StackingEnsemble             # noqa: F401

import joblib

# ── Defaults ─────────────────────────────────────────────────────
DEFAULT_ENSEMBLE = ROOT / "models" / "final_ensemble.joblib"
DEFAULT_INPUT = ROOT / "examples" / "sample_input.json"
DEFAULT_OUTPUT = ROOT / "examples" / "sample_output.json"
VIOLATION_THRESHOLD = 0.5


# ─────────────────────────────────────────────────────────────────
# Core functions
# ─────────────────────────────────────────────────────────────────

def load_ensemble(path: pathlib.Path = DEFAULT_ENSEMBLE) -> Dict[str, Any]:
    """Load the saved ensemble artefact (reg, clf, transformer)."""
    ens = joblib.load(path)
    return ens


def read_input(path: pathlib.Path) -> pd.DataFrame:
    """Read a JSON file with ``{"flows": [...]}`` and return a DataFrame."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    flows = raw.get("flows", raw if isinstance(raw, list) else [raw])
    df = pd.DataFrame(flows)
    return df


def predict(
    df: pd.DataFrame,
    ensemble: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Run predictions on a DataFrame of flow records.

    Returns a list of dicts with per-flow predictions.
    """
    ct = ensemble["transformer"]
    ens_reg = ensemble["reg"]
    ens_clf = ensemble["clf"]

    # Transform features
    X = ct.transform(df)

    # Regression – predicted latency in μs
    lat_pred = ens_reg.predict(X)

    # Classification – probability of violation
    prob = ens_clf.predict_proba(X)[:, 1]

    results: List[Dict[str, Any]] = []
    for i in range(len(df)):
        results.append(
            {
                "flow_index": i,
                "predicted_latency_us": round(float(lat_pred[i]), 4),
                "violation_probability": round(float(prob[i]), 6),
                "violation_flag": bool(prob[i] >= VIOLATION_THRESHOLD),
            }
        )
    return results


def write_output(predictions: List[Dict[str, Any]], path: pathlib.Path) -> None:
    """Write predictions to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"predictions": predictions}, f, indent=2)
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Online prediction with the stacking ensemble.",
    )
    parser.add_argument(
        "--input", "-i",
        type=pathlib.Path,
        default=DEFAULT_INPUT,
        help="Path to input JSON batch file.",
    )
    parser.add_argument(
        "--output", "-o",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT,
        help="Path to write output JSON.",
    )
    parser.add_argument(
        "--model", "-m",
        type=pathlib.Path,
        default=DEFAULT_ENSEMBLE,
        help="Path to final_ensemble.joblib.",
    )
    args = parser.parse_args(argv)

    print("Loading ensemble …")
    ens = load_ensemble(args.model)

    print(f"Reading input: {args.input}")
    df = read_input(args.input)
    print(f"  {len(df)} flow(s)")

    print("Running predictions …")
    preds = predict(df, ens)

    write_output(preds, args.output)

    # Quick summary
    for p in preds:
        flag = "VIOLATION" if p["violation_flag"] else "ok"
        print(
            f"  flow {p['flow_index']}: "
            f"latency={p['predicted_latency_us']:.2f} μs  "
            f"P(violation)={p['violation_probability']:.4f}  [{flag}]"
        )

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
