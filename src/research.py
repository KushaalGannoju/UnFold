"""Read actual model metadata and optional paired evaluation artifacts."""

import json
from pathlib import Path

EVALUATION_PATH = Path(__file__).resolve().parents[1] / "research" / "hybrid_evaluation.json"


def benchmark_rows(bundle):
    comparison = bundle.get("model_comparison_df")
    rows = []
    if comparison is not None:
        for item in comparison.to_dict("records"):
            rows.append({"Model": item["model"], "Evaluation": "Saved tabular test split",
                         "Accuracy": item["test_accuracy"], "Precision": item["precision"],
                         "Recall": item["recall"], "F1": item["f1_score"]})
    return rows


def load_paired_evaluation(path=EVALUATION_PATH):
    if not Path(path).exists():
        return None
    with open(path) as source:
        data = json.load(source)
    if data.get("schema_version") != 1 or not data.get("sample_count") or not data.get("metrics"):
        raise ValueError("The paired evaluation artifact is incomplete.")
    return data
