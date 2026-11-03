"""Evaluate all three scores on a genuinely paired, held-out CSV. Never sends data online."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, brier_score_loss, confusion_matrix
from assessment import DEFAULT_PROFILE, validate_profile
from predict import predict
from nlp_engine import predict_text
from fusion import fuse_predictions


def evaluate_paired(frame):
    required = set(DEFAULT_PROFILE) | {"reflection", "depression"}
    if not required <= set(frame.columns) or frame.empty:
        raise ValueError("CSV requires all canonical profile columns, reflection, and depression (0 or 1).")
    if frame[list(required)].isna().any().any() or not set(frame.depression.unique()) <= {0, 1}:
        raise ValueError("Evaluation data contains missing values or invalid labels.")
    if frame.depression.nunique() != 2:
        raise ValueError("Both labeled classes are required for evaluation.")
    scores = {"Profile GBDT": [], "Journal RoBERTa": [], "Combined": []}
    for record in frame.to_dict("records"):
        profile = {k: record[k] for k in DEFAULT_PROFILE}
        validate_profile(profile)
        if not isinstance(record["reflection"], str) or not record["reflection"].strip():
            raise ValueError("Every paired sample must have a non-empty reflection.")
        tab, text = predict(profile), predict_text(record["reflection"])
        fused = fuse_predictions(tab, text)
        for name, score in zip(scores, [tab["prob_dep"], text["prob_dep"], fused["hybrid_prob_dep"]]):
            scores[name].append(score / 100)
    metrics = {}
    for name, values in scores.items():
        predictions = [int(p >= 0.5) for p in values]
        precision, recall, f1, _ = precision_recall_fscore_support(frame.depression, predictions, average="binary", zero_division=0)
        metrics[name] = {"accuracy": accuracy_score(frame.depression, predictions), "precision": precision,
                         "recall": recall, "f1": f1, "brier_score": brier_score_loss(frame.depression, values),
                         "confusion_matrix": confusion_matrix(frame.depression, predictions, labels=[0, 1]).tolist()}
    return {"schema_version": 1, "sample_count": len(frame), "metrics": metrics,
            "evaluated_at": datetime.now(timezone.utc).isoformat(), "calibration": "Not fitted"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split-description", required=True, help="Describe how these students were held out from both models.")
    parser.add_argument("--output", default="research/hybrid_evaluation.json")
    args = parser.parse_args()
    result = evaluate_paired(pd.read_csv(args.data))
    result["split_description"] = args.split_description
    result["dataset_sha256"] = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(f"Saved aggregate metrics for {result['sample_count']} paired samples to {output}.")


if __name__ == "__main__":
    main()
