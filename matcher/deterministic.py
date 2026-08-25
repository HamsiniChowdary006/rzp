from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from typing import Iterable

import pandas as pd
from rapidfuzz import fuzz

from schemas import (
    BankStatementRow,
    ExceptionRecord,
    ExceptionSeverity,
    GatewaySettlementRow,
    InternalLedgerRow,
    MatchStatus,
    MatchVerdict,
    PatternDetected,
    ReconciliationSummary,
    quantize_money,
)


PREFIX_UTR_LENGTH = 10
AUTO_MATCH_CONFIDENCE = 0.99
AMBIGUOUS_CONFIDENCE = 0.84
STANDARD_FEE_RATE = Decimal("0.02")
STANDARD_TAX_RATE = Decimal("0.18")


@dataclass(frozen=True)
class ReconciliationResult:
    matches: list[MatchVerdict]
    verifier_candidates: list[MatchVerdict]
    exceptions: list[ExceptionRecord]
    summary: ReconciliationSummary


class DeterministicMatchEngine:
    """Rule-first reconciliation engine with honest ambiguity routing."""

    def __init__(self, date_window_days: int = 1) -> None:
        self.date_window_days = date_window_days

    def reconcile(
        self,
        gateway_rows: pd.DataFrame,
        bank_rows: pd.DataFrame,
        ledger_rows: pd.DataFrame,
    ) -> ReconciliationResult:
        start = perf_counter()
        gateway = self._prepare_gateway(gateway_rows)
        bank = self._prepare_bank(bank_rows)
        ledger = self._prepare_ledger(ledger_rows)

        settlement_chains = self._build_settlement_chains(gateway, ledger)
        bank_candidate_map = self._candidate_bank_rows(bank, settlement_chains)

        matches: list[MatchVerdict] = []
        verifier_candidates: list[MatchVerdict] = []
        exceptions: list[ExceptionRecord] = []
        used_bank_txn_ids: set[str] = set()

        for _, chain in settlement_chains.iterrows():
            candidates = bank_candidate_map.get(chain.settlement_id, pd.DataFrame())
            candidates = candidates.sort_values(["txn_id"]) if not candidates.empty else candidates

            if candidates.empty:
                verifier_candidates.append(self._candidate_for_missing_bank_row(chain))
                continue

            exact_candidates = [
                bank_row
                for _, bank_row in candidates.iterrows()
                if self._is_exact_clean_match(chain, bank_row)
            ]

            if len(exact_candidates) == 1:
                bank_row = exact_candidates[0]
                matches.append(self._clean_match(chain, bank_row))
                used_bank_txn_ids.add(bank_row.txn_id)
                continue

            if len(exact_candidates) > 1:
                primary_bank_row = sorted(exact_candidates, key=lambda row: row.txn_id)[0]
                matches.append(self._clean_match(chain, primary_bank_row))
                used_bank_txn_ids.add(primary_bank_row.txn_id)
                continue

            best_bank_row = candidates.iloc[0]
            verifier_candidates.append(self._classify_candidate(chain, best_bank_row))

        exceptions.extend(self._duplicate_exceptions(bank, used_bank_txn_ids))
        exceptions.extend(self._unmatched_bank_exceptions(bank, used_bank_txn_ids, bank_candidate_map))

        elapsed_ms = (perf_counter() - start) * 1000
        summary = ReconciliationSummary(
            total_gateway_rows=len(gateway),
            total_bank_rows=len(bank),
            total_ledger_rows=len(ledger),
            deterministic_matches=len(matches),
            verifier_candidates=len(verifier_candidates),
            exceptions=len(exceptions),
            elapsed_ms=elapsed_ms,
        )
        return ReconciliationResult(
            matches=matches,
            verifier_candidates=verifier_candidates,
            exceptions=exceptions,
            summary=summary,
        )

    @classmethod
    def from_csv(
        cls,
        gateway_path: str | Path,
        bank_path: str | Path,
        ledger_path: str | Path,
    ) -> ReconciliationResult:
        engine = cls()
        return engine.reconcile(
            pd.read_csv(gateway_path),
            pd.read_csv(bank_path),
            pd.read_csv(ledger_path),
        )

    def _prepare_gateway(self, rows: pd.DataFrame) -> pd.DataFrame:
        validated = [GatewaySettlementRow.model_validate(row) for row in rows.to_dict("records")]
        frame = pd.DataFrame([row.model_dump() for row in validated])
        frame["settled_at"] = pd.to_datetime(frame["settled_at"]).dt.date
        frame["net_amount_calc"] = frame.apply(
            lambda row: quantize_money(row["amount"] - row["fee"] - row["tax"]), axis=1
        )
        return frame

    def _prepare_bank(self, rows: pd.DataFrame) -> pd.DataFrame:
        validated = [BankStatementRow.model_validate(row) for row in rows.to_dict("records")]
        frame = pd.DataFrame([row.model_dump() for row in validated])
        frame["value_date"] = pd.to_datetime(frame["value_date"]).dt.date
        frame["unsafe_narration"] = frame["narration"].astype(str)
        return frame

    def _prepare_ledger(self, rows: pd.DataFrame) -> pd.DataFrame:
        validated = [InternalLedgerRow.model_validate(row) for row in rows.to_dict("records")]
        frame = pd.DataFrame([row.model_dump() for row in validated])
        frame["invoice_date"] = pd.to_datetime(frame["invoice_date"]).dt.date
        return frame

    def _build_settlement_chains(self, gateway: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
        chains = gateway.merge(ledger, on="order_id", how="left", validate="one_to_one")
        missing_ledger = chains[chains["ledger_id"].isna()]
        if not missing_ledger.empty:
            missing_ids = ", ".join(missing_ledger["settlement_id"].astype(str).tolist())
            raise ValueError(f"gateway rows without ledger counterpart: {missing_ids}")
        return chains

    def _candidate_bank_rows(
        self,
        bank: pd.DataFrame,
        settlement_chains: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        candidates: dict[str, pd.DataFrame] = {}
        for _, chain in settlement_chains.iterrows():
            exact_utr = bank["utr_ref"] == chain.utr
            prefix_utr = bank["utr_ref"].astype(str).str[:PREFIX_UTR_LENGTH] == str(chain.utr)[:PREFIX_UTR_LENGTH]
            same_net_amount = bank["amount"] == chain.net_amount
            nearby_amount = bank["amount"].map(lambda amount: abs(Decimal(amount) - chain.net_amount) <= Decimal("0.01"))

            candidate_rows = bank[exact_utr | (prefix_utr & (same_net_amount | nearby_amount))].copy()
            if not candidate_rows.empty:
                candidate_rows["candidate_score"] = candidate_rows.apply(
                    lambda row: self._candidate_score(chain, row), axis=1
                )
                candidates[chain.settlement_id] = candidate_rows.sort_values(
                    ["candidate_score", "txn_id"], ascending=[False, True]
                )
        return candidates

    def _candidate_score(self, chain: pd.Series, bank_row: pd.Series) -> float:
        utr_score = fuzz.ratio(str(chain.utr), str(bank_row.utr_ref)) / 100
        amount_score = 1.0 if bank_row.amount == chain.net_amount else 0.0
        date_score = 1.0 if self._date_delta(chain.invoice_date, bank_row.value_date) <= self.date_window_days else 0.4
        return round((0.55 * utr_score) + (0.35 * amount_score) + (0.10 * date_score), 4)

    def _is_exact_clean_match(self, chain: pd.Series, bank_row: pd.Series) -> bool:
        return (
            str(chain.utr) == str(bank_row.utr_ref)
            and chain.net_amount == bank_row.amount
            and chain.net_amount_calc == chain.net_amount
            and self._has_standard_fee_schedule(chain)
            and chain.amount == chain.expected_amount
            and self._date_delta(chain.invoice_date, bank_row.value_date) <= self.date_window_days
        )

    def _clean_match(self, chain: pd.Series, bank_row: pd.Series) -> MatchVerdict:
        return MatchVerdict(
            status=MatchStatus.MATCH,
            match=True,
            matched_settlement_id=chain.settlement_id,
            matched_ledger_id=chain.ledger_id,
            matched_bank_txn_id=bank_row.txn_id,
            confidence=AUTO_MATCH_CONFIDENCE,
            pattern_detected=PatternDetected.CLEAN_MATCH,
            reasoning=(
                "Exact 3-way match: ledger gross equals gateway gross, gateway net "
                "reconciles to gross minus fee and tax, bank UTR/net amount match, "
                "and value date is within the standard T+1 invoice window."
            ),
            evidence_fields_used=("order_id", "utr", "net_amount", "fee", "tax", "invoice_date", "value_date"),
            requires_verifier=False,
        )

    def _classify_candidate(self, chain: pd.Series, bank_row: pd.Series) -> MatchVerdict:
        utr_exact = str(chain.utr) == str(bank_row.utr_ref)
        utr_prefix = str(chain.utr).startswith(str(bank_row.utr_ref)[:PREFIX_UTR_LENGTH])
        amount_exact = chain.net_amount == bank_row.amount
        date_delta = self._date_delta(chain.invoice_date, bank_row.value_date)

        if "ignore all previous instructions" in str(bank_row.unsafe_narration).lower():
            return self._ambiguous_candidate(
                chain,
                bank_row,
                PatternDetected.AMOUNT_MISMATCH_WITH_INJECTION_ATTEMPT,
                "Bank narration contains prompt-injection text and amount does not reconcile exactly.",
            )
        if utr_exact and chain.net_amount_calc != chain.net_amount:
            return self._ambiguous_candidate(
                chain,
                bank_row,
                PatternDetected.PARTIAL_REFUND,
                "Gateway net amount differs from gross minus fee and tax, indicating refund or adjustment handling.",
            )
        if utr_exact and bank_row.amount < chain.net_amount:
            return self._ambiguous_candidate(
                chain,
                bank_row,
                PatternDetected.PARTIAL_REFUND,
                "UTR chain matches, but bank credit is lower than gateway net amount.",
            )
        if utr_prefix and not utr_exact and amount_exact:
            return self._ambiguous_candidate(
                chain,
                bank_row,
                PatternDetected.TRUNCATED_UTR,
                "Bank UTR appears truncated; prefix and net amount match but exact UTR is unavailable.",
            )
        if utr_exact and amount_exact and date_delta > self.date_window_days:
            return self._ambiguous_candidate(
                chain,
                bank_row,
                PatternDetected.TIMING_MISMATCH,
                "UTR and net amount match, but value date is outside the standard T+1 invoice window.",
            )
        if utr_exact and amount_exact and chain.amount == chain.expected_amount:
            return self._ambiguous_candidate(
                chain,
                bank_row,
                PatternDetected.FEE_SPLIT_MISMATCH,
                "Gateway fee/tax calculation reconciles gross ledger amount to bank net amount, requiring verifier attestation.",
            )
        return self._ambiguous_candidate(
            chain,
            bank_row,
            PatternDetected.UNKNOWN,
            "Candidate shares at least one deterministic key but does not satisfy auto-match rules.",
            )

    def _candidate_for_missing_bank_row(self, chain: pd.Series) -> MatchVerdict:
        return MatchVerdict(
            status=MatchStatus.NEEDS_VERIFIER,
            match=False,
            matched_settlement_id=chain.settlement_id,
            matched_ledger_id=chain.ledger_id,
            matched_bank_txn_id=None,
            confidence=0.0,
            pattern_detected=PatternDetected.UNMATCHABLE,
            reasoning="No bank transaction shares the gateway UTR or a supported truncated UTR prefix.",
            evidence_fields_used=("order_id", "utr", "net_amount"),
            requires_verifier=True,
        )

    def _ambiguous_candidate(
        self,
        chain: pd.Series,
        bank_row: pd.Series,
        pattern: PatternDetected,
        reason: str,
    ) -> MatchVerdict:
        return MatchVerdict(
            status=MatchStatus.NEEDS_VERIFIER,
            match=False,
            matched_settlement_id=chain.settlement_id,
            matched_ledger_id=chain.ledger_id,
            matched_bank_txn_id=bank_row.txn_id,
            confidence=AMBIGUOUS_CONFIDENCE,
            pattern_detected=pattern,
            reasoning=reason,
            evidence_fields_used=("order_id", "utr", "utr_prefix", "net_amount", "fee", "tax", "invoice_date", "value_date"),
            requires_verifier=True,
        )

    def _duplicate_exceptions(
        self,
        bank: pd.DataFrame,
        used_bank_txn_ids: set[str],
    ) -> list[ExceptionRecord]:
        exceptions: list[ExceptionRecord] = []
        duplicate_groups = bank[bank.duplicated(["utr_ref", "amount", "value_date"], keep=False)]
        for _, group in duplicate_groups.groupby(["utr_ref", "amount", "value_date"], sort=False):
            txn_ids = sorted(group["txn_id"].tolist())
            for txn_id in txn_ids[1:]:
                if txn_id in used_bank_txn_ids:
                    continue
                exceptions.append(
                    ExceptionRecord(
                        exception_id=f"EXC-{txn_id}",
                        bank_txn_id=txn_id,
                        root_cause=PatternDetected.DUPLICATE_BANK_ENTRY,
                        severity=ExceptionSeverity.HIGH,
                        confidence=1.0,
                        reason="Duplicate bank credit for an already consumed UTR/amount/date group.",
                        evidence={"duplicate_of": txn_ids[0], "utr_ref": str(group.iloc[0].utr_ref)},
                    )
                )
        return exceptions

    def _unmatched_bank_exceptions(
        self,
        bank: pd.DataFrame,
        used_bank_txn_ids: set[str],
        bank_candidate_map: dict[str, pd.DataFrame],
    ) -> list[ExceptionRecord]:
        candidate_txn_ids = {
            txn_id
            for candidate_rows in bank_candidate_map.values()
            for txn_id in candidate_rows["txn_id"].tolist()
        }
        exceptions: list[ExceptionRecord] = []
        for _, bank_row in bank.iterrows():
            if bank_row.txn_id in used_bank_txn_ids or bank_row.txn_id in candidate_txn_ids:
                continue
            exceptions.append(
                ExceptionRecord(
                    exception_id=f"EXC-{bank_row.txn_id}",
                    bank_txn_id=bank_row.txn_id,
                    root_cause=PatternDetected.UNMATCHABLE,
                    severity=ExceptionSeverity.MEDIUM,
                    confidence=1.0,
                    reason="No gateway settlement or ledger candidate shares this bank UTR or supported UTR prefix.",
                    evidence={"utr_ref": str(bank_row.utr_ref), "amount": float(bank_row.amount)},
                )
            )
        return exceptions

    @staticmethod
    def _date_delta(left: object, right: object) -> int:
        left_date = pd.to_datetime(left).date()
        right_date = pd.to_datetime(right).date()
        return abs((right_date - left_date).days)

    @staticmethod
    def _has_standard_fee_schedule(chain: pd.Series) -> bool:
        expected_fee = quantize_money(chain.amount * STANDARD_FEE_RATE)
        expected_tax = quantize_money(expected_fee * STANDARD_TAX_RATE)
        return chain.fee == expected_fee and chain.tax == expected_tax


def reconcile(
    gateway_rows: pd.DataFrame,
    bank_rows: pd.DataFrame,
    ledger_rows: pd.DataFrame,
) -> ReconciliationResult:
    return DeterministicMatchEngine().reconcile(gateway_rows, bank_rows, ledger_rows)
