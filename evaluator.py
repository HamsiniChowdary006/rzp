from __future__ import annotations

from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Literal

import pandas as pd
from pydantic import Field

from agent.graph import run_reconciliation_from_csv
from schemas import Confidence, PatternDetected, StrictBaseModel


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


class ClassificationMetrics(StrictBaseModel):
    precision: Confidence
    recall: Confidence
    f1_score: Confidence
    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)


class PatternAccuracy(StrictBaseModel):
    pattern: PatternDetected
    expected_count: int = Field(ge=0)
    correct_count: int = Field(ge=0)
    accuracy: Confidence


class ThroughputMetrics(StrictBaseModel):
    total_execution_time_ms: float = Field(ge=0.0)
    records_processed: int = Field(ge=0)
    records_per_second: float = Field(ge=0.0)


class EvaluationReport(StrictBaseModel):
    matches: ClassificationMetrics
    exceptions: ClassificationMetrics
    overall_accuracy: Confidence
    pattern_accuracy: dict[str, PatternAccuracy]
    throughput: ThroughputMetrics
    total_predictions: int = Field(ge=0)
    total_ground_truth: int = Field(ge=0)


def evaluate_default_dataset(data_dir: Path = DEFAULT_DATA_DIR) -> EvaluationReport:
    return evaluate_reconciliation(
        gateway_path=data_dir / "gateway_settlements.csv",
        bank_path=data_dir / "bank_statement.csv",
        ledger_path=data_dir / "internal_ledger.csv",
        ground_truth_path=data_dir / "ground_truth.csv",
    )


def evaluate_reconciliation(
    *,
    gateway_path: str | Path,
    bank_path: str | Path,
    ledger_path: str | Path,
    ground_truth_path: str | Path,
) -> EvaluationReport:
    started_at = perf_counter()
    state = run_reconciliation_from_csv(gateway_path, bank_path, ledger_path)
    elapsed_ms = (perf_counter() - started_at) * 1000
    ground_truth = pd.read_csv(ground_truth_path).fillna("")

    predicted = _predicted_rows(state)
    expected = _expected_rows(ground_truth)
    match_metrics = _classification_metrics(predicted, expected, label="match")
    exception_metrics = _classification_metrics(predicted, expected, label="exception")
    correct_predictions = sum(1 for key, value in predicted.items() if expected.get(key) == value)
    total_records = len(expected)

    return EvaluationReport(
        matches=match_metrics,
        exceptions=exception_metrics,
        overall_accuracy=_safe_ratio(correct_predictions, total_records),
        pattern_accuracy=_pattern_accuracy(predicted, expected),
        throughput=ThroughputMetrics(
            total_execution_time_ms=round(elapsed_ms, 3),
            records_processed=total_records,
            records_per_second=round(total_records / (elapsed_ms / 1000), 3) if elapsed_ms else 0.0,
        ),
        total_predictions=len(predicted),
        total_ground_truth=total_records,
    )


def _predicted_rows(state: dict) -> dict[str, tuple[str, str]]:
    rows: dict[str, tuple[str, str]] = {}
    for match in state.get("matches", []):
        if match.matched_bank_txn_id is None:
            continue
        rows[match.matched_bank_txn_id] = ("match", match.pattern_detected.value)
    for exception in state.get("exceptions", []):
        if exception.bank_txn_id is None:
            continue
        rows[exception.bank_txn_id] = ("exception", exception.root_cause.value)
    return rows


def _expected_rows(ground_truth: pd.DataFrame) -> dict[str, tuple[str, str]]:
    return {
        str(row.bank_txn_id): (str(row.label), str(row.pattern))
        for row in ground_truth.itertuples(index=False)
        if str(row.bank_txn_id)
    }


def _classification_metrics(
    predicted: dict[str, tuple[str, str]],
    expected: dict[str, tuple[str, str]],
    *,
    label: Literal["match", "exception"],
) -> ClassificationMetrics:
    predicted_ids = {txn_id for txn_id, outcome in predicted.items() if outcome[0] == label}
    expected_ids = {txn_id for txn_id, outcome in expected.items() if outcome[0] == label}
    true_positives = len(predicted_ids & expected_ids)
    false_positives = len(predicted_ids - expected_ids)
    false_negatives = len(expected_ids - predicted_ids)
    precision = _safe_ratio(true_positives, true_positives + false_positives)
    recall = _safe_ratio(true_positives, true_positives + false_negatives)
    f1_score = (
        0.0
        if precision + recall == 0
        else round((2 * precision * recall) / (precision + recall), 6)
    )
    return ClassificationMetrics(
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def _pattern_accuracy(
    predicted: dict[str, tuple[str, str]],
    expected: dict[str, tuple[str, str]],
) -> dict[str, PatternAccuracy]:
    expected_counts = Counter(pattern for _, pattern in expected.values())
    correct_counts = Counter(
        expected_outcome[1]
        for txn_id, expected_outcome in expected.items()
        if predicted.get(txn_id) == expected_outcome
    )
    return {
        pattern.value: PatternAccuracy(
            pattern=pattern,
            expected_count=expected_counts[pattern.value],
            correct_count=correct_counts[pattern.value],
            accuracy=_safe_ratio(correct_counts[pattern.value], expected_counts[pattern.value]),
        )
        for pattern in PatternDetected
        if pattern != PatternDetected.UNKNOWN and expected_counts[pattern.value] > 0
    }


def _safe_ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0
