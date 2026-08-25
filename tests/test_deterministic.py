from __future__ import annotations

from pathlib import Path

import pandas as pd

from matcher.deterministic import DeterministicMatchEngine
from schemas import PatternDetected


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_clean_matches_have_100_percent_precision_against_ground_truth() -> None:
    result = DeterministicMatchEngine().from_csv(
        DATA_DIR / "gateway_settlements.csv",
        DATA_DIR / "bank_statement.csv",
        DATA_DIR / "internal_ledger.csv",
    )
    ground_truth = pd.read_csv(DATA_DIR / "ground_truth.csv")
    clean_truth = ground_truth[ground_truth["pattern"] == PatternDetected.CLEAN_MATCH.value]
    expected = {
        (row.ledger_id, row.settlement_id, row.bank_txn_id)
        for row in clean_truth.itertuples(index=False)
    }
    actual = {
        (
            match.matched_ledger_id,
            match.matched_settlement_id,
            match.matched_bank_txn_id,
        )
        for match in result.matches
    }

    assert len(actual) == 33
    assert actual == expected


def test_ambiguous_rows_are_not_forced_into_matches() -> None:
    result = DeterministicMatchEngine().from_csv(
        DATA_DIR / "gateway_settlements.csv",
        DATA_DIR / "bank_statement.csv",
        DATA_DIR / "internal_ledger.csv",
    )
    matched_bank_txn_ids = {match.matched_bank_txn_id for match in result.matches}

    assert "TXN0050" not in matched_bank_txn_ids
    assert all(match.confidence >= 0.85 for match in result.matches)
    assert all(candidate.requires_verifier for candidate in result.verifier_candidates)


def test_known_exception_patterns_are_identified_for_human_review() -> None:
    result = DeterministicMatchEngine().from_csv(
        DATA_DIR / "gateway_settlements.csv",
        DATA_DIR / "bank_statement.csv",
        DATA_DIR / "internal_ledger.csv",
    )
    candidate_patterns = {
        candidate.matched_bank_txn_id: candidate.pattern_detected
        for candidate in result.verifier_candidates
        if candidate.matched_bank_txn_id is not None
    }
    exception_patterns = {
        exception.bank_txn_id: exception.root_cause
        for exception in result.exceptions
        if exception.bank_txn_id is not None
    }

    assert candidate_patterns["TXN0049"] == PatternDetected.PARTIAL_REFUND
    assert candidate_patterns["TXN0050"] == PatternDetected.AMOUNT_MISMATCH_WITH_INJECTION_ATTEMPT
    assert exception_patterns["TXN0048"] == PatternDetected.DUPLICATE_BANK_ENTRY
    assert exception_patterns["TXN0051"] == PatternDetected.UNMATCHABLE
    assert exception_patterns["TXN0052"] == PatternDetected.UNMATCHABLE
    assert exception_patterns["TXN0053"] == PatternDetected.UNMATCHABLE


def test_gateway_gross_fee_tax_net_formula_is_enforced() -> None:
    result = DeterministicMatchEngine().from_csv(
        DATA_DIR / "gateway_settlements.csv",
        DATA_DIR / "bank_statement.csv",
        DATA_DIR / "internal_ledger.csv",
    )

    assert result.summary.total_gateway_rows == 49
    assert result.summary.total_bank_rows == 53
    assert result.summary.total_ledger_rows == 49
    assert result.summary.deterministic_matches == 33
