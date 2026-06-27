from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_attribution_map(attribution_map: np.ndarray) -> np.ndarray:
    values = np.asarray(attribution_map, dtype=np.float64)
    values = values - values.min()

    max_value = values.max()
    if max_value <= 0:
        return np.zeros_like(values)

    return values / max_value


def weighted_jaccard(first_map: np.ndarray, second_map: np.ndarray) -> float:
    first_values = normalize_attribution_map(first_map).ravel()
    second_values = normalize_attribution_map(second_map).ravel()

    denominator = np.maximum(first_values, second_values).sum()
    if denominator <= 0:
        return 0.0

    numerator = np.minimum(first_values, second_values).sum()
    return float(numerator / denominator)


def dad(first_map: np.ndarray, second_map: np.ndarray) -> float:
    return 1.0 - weighted_jaccard(first_map, second_map)


def bootstrap_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0

    random_generator = np.random.default_rng(seed)
    bootstrap_values = []

    for _ in range(n_bootstrap):
        sample = random_generator.choice(values, size=values.size, replace=True)
        bootstrap_values.append(float(sample.mean()))

    alpha = 1.0 - confidence
    lower = np.quantile(bootstrap_values, alpha / 2.0)
    upper = np.quantile(bootstrap_values, 1.0 - alpha / 2.0)

    return float(lower), float(upper)


def summarize_records(records_data: pd.DataFrame) -> dict[str, float | int]:
    same_prediction = records_data[records_data["same_prediction"] == 1]
    both_correct = records_data[
        (records_data["neat_pure_correct"] == 1)
        & (records_data["neat_gradient_correct"] == 1)
    ]
    both_wrong = records_data[
        (records_data["neat_pure_correct"] == 0)
        & (records_data["neat_gradient_correct"] == 0)
    ]
    same_wrong_prediction = both_wrong[
        both_wrong["neat_pure_prediction"]
        == both_wrong["neat_gradient_prediction"]
    ]

    dad_values = records_data["dad"].to_numpy(dtype=np.float64)
    dad_lower, dad_upper = bootstrap_ci(dad_values)

    raw_gap = (
        float(same_prediction["weighted_jaccard"].mean())
        - float(both_wrong["weighted_jaccard"].mean())
        if len(same_prediction) > 0 and len(both_wrong) > 0
        else 0.0
    )

    return {
        "n_records": int(len(records_data)),
        "global_weighted_jaccard": float(records_data["weighted_jaccard"].mean()),
        "global_dad": float(records_data["dad"].mean()),
        "global_dad_ci95_lower": dad_lower,
        "global_dad_ci95_upper": dad_upper,
        "prediction_agreement": float(records_data["same_prediction"].mean()),
        "same_prediction_jaccard": float(same_prediction["weighted_jaccard"].mean())
        if len(same_prediction) > 0
        else 0.0,
        "both_correct_jaccard": float(both_correct["weighted_jaccard"].mean())
        if len(both_correct) > 0
        else 0.0,
        "both_wrong_jaccard": float(both_wrong["weighted_jaccard"].mean())
        if len(both_wrong) > 0
        else 0.0,
        "same_wrong_prediction_jaccard": float(
            same_wrong_prediction["weighted_jaccard"].mean()
        )
        if len(same_wrong_prediction) > 0
        else 0.0,
        "raw_gap": raw_gap,
        "n_same_prediction": int(len(same_prediction)),
        "n_both_correct": int(len(both_correct)),
        "n_both_wrong": int(len(both_wrong)),
        "n_same_wrong_prediction": int(len(same_wrong_prediction)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records_path = args.artifact_dir / "interpretability_records.csv"

    if not records_path.exists():
        raise FileNotFoundError(
            "Arquivo interpretability_records.csv não encontrado. "
            "Primeiro gere os mapas e registros por amostra."
        )

    records_data = pd.read_csv(records_path)
    metrics = summarize_records(records_data)

    output_path = args.artifact_dir / "interpretability_metrics_extended.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    print(f"Métricas salvas em: {output_path}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()