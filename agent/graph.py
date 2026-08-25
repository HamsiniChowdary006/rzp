from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypedDict

import pandas as pd

from agent.verifier import LLMVerifier
from matcher.deterministic import DeterministicMatchEngine
from schemas import (
    AgentVerificationResult,
    ExceptionRecord,
    ExceptionSeverity,
    MatchStatus,
    MatchVerdict,
    PatternDetected,
    ReconciliationSummary,
)


class ReconciliationState(TypedDict, total=False):
    gateway_rows: pd.DataFrame
    bank_rows: pd.DataFrame
    ledger_rows: pd.DataFrame
    unmatched_bank_rows: list[dict[str, Any]]
    candidate_pool: list[dict[str, Any]]
    verifier_queue: list[MatchVerdict]
    verifier_results: list[tuple[MatchVerdict, AgentVerificationResult]]
    matches: list[MatchVerdict]
    exceptions: list[ExceptionRecord]
    summary: ReconciliationSummary


@dataclass
class RunnableReconciliationGraph:
    nodes: list[str]
    runner: Callable[[ReconciliationState], ReconciliationState]

    def invoke(self, state: ReconciliationState) -> ReconciliationState:
        return self.runner(state)


class ReconciliationGraphBuilder:
    def __init__(
        self,
        deterministic_engine: DeterministicMatchEngine | None = None,
        verifier: LLMVerifier | None = None,
    ) -> None:
        self.deterministic_engine = deterministic_engine or DeterministicMatchEngine()
        self.verifier = verifier or LLMVerifier()

    def build(self) -> Any:
        try:
            from langgraph.graph import END, StateGraph

            graph = StateGraph(ReconciliationState)
            graph.add_node("Ingest", self.ingest)
            graph.add_node("DeterministicPass", self.deterministic_pass)
            graph.add_node("RouteAmbiguous", self.route_ambiguous)
            graph.add_node("LLMVerifierNode", self.llm_verifier_node)
            graph.add_node("DecisionGateNode", self.decision_gate_node)
            graph.add_node("Aggregate", self.aggregate)
            graph.set_entry_point("Ingest")
            graph.add_edge("Ingest", "DeterministicPass")
            graph.add_edge("DeterministicPass", "RouteAmbiguous")
            graph.add_edge("RouteAmbiguous", "LLMVerifierNode")
            graph.add_edge("LLMVerifierNode", "DecisionGateNode")
            graph.add_edge("DecisionGateNode", "Aggregate")
            graph.add_edge("Aggregate", END)
            return graph.compile()
        except Exception:
            return RunnableReconciliationGraph(
                nodes=[
                    "Ingest",
                    "DeterministicPass",
                    "RouteAmbiguous",
                    "LLMVerifierNode",
                    "DecisionGateNode",
                    "Aggregate",
                ],
                runner=self._run_fallback,
            )

    def _run_fallback(self, state: ReconciliationState) -> ReconciliationState:
        for node in (
            self.ingest,
            self.deterministic_pass,
            self.route_ambiguous,
            self.llm_verifier_node,
            self.decision_gate_node,
            self.aggregate,
        ):
            state = node(state)
        return state

    def ingest(self, state: ReconciliationState) -> ReconciliationState:
        required = {"gateway_rows", "bank_rows", "ledger_rows"}
        missing = sorted(required - set(state))
        if missing:
            raise ValueError(f"missing graph input(s): {', '.join(missing)}")
        return state

    def deterministic_pass(self, state: ReconciliationState) -> ReconciliationState:
        result = self.deterministic_engine.reconcile(
            state["gateway_rows"],
            state["bank_rows"],
            state["ledger_rows"],
        )
        return state | {
            "matches": list(result.matches),
            "exceptions": list(result.exceptions),
            "verifier_queue": list(result.verifier_candidates),
        }

    def route_ambiguous(self, state: ReconciliationState) -> ReconciliationState:
        matches = state.get("matches", [])
        used_settlement_ids = {
            match.matched_settlement_id
            for match in matches
            if match.matched_settlement_id is not None
        }
        used_bank_txn_ids = {
            match.matched_bank_txn_id
            for match in matches
            if match.matched_bank_txn_id is not None
        }
        candidate_pool = self._candidate_pool(
            state["gateway_rows"],
            state["ledger_rows"],
            excluded_settlement_ids=used_settlement_ids,
        )
        unmatched_bank_rows = self._unmatched_bank_rows(
            state["bank_rows"],
            excluded_bank_txn_ids=used_bank_txn_ids,
        )
        return state | {
            "candidate_pool": candidate_pool,
            "unmatched_bank_rows": unmatched_bank_rows,
        }

    def llm_verifier_node(self, state: ReconciliationState) -> ReconciliationState:
        bank_by_txn_id = {
            str(row["txn_id"]): row
            for row in state.get("unmatched_bank_rows", [])
        }
        candidate_pool = state.get("candidate_pool", [])
        verifier_results: list[tuple[MatchVerdict, AgentVerificationResult]] = []
        for candidate_verdict in state.get("verifier_queue", []):
            bank_txn_id = candidate_verdict.matched_bank_txn_id
            if bank_txn_id is None or bank_txn_id not in bank_by_txn_id:
                continue
            preferred_pool = self._prioritise_declared_candidate(candidate_verdict, candidate_pool)
            verifier_result = self.verifier.verify_bank_row(
                bank_by_txn_id[bank_txn_id],
                preferred_pool,
                top_k=3,
            )
            verifier_results.append((candidate_verdict, verifier_result))
        return state | {"verifier_results": verifier_results}

    def decision_gate_node(self, state: ReconciliationState) -> ReconciliationState:
        matches = list(state.get("matches", []))
        exceptions = list(state.get("exceptions", []))
        matched_bank_ids = {
            match.matched_bank_txn_id
            for match in matches
            if match.matched_bank_txn_id is not None
        }

        for candidate_verdict, verifier_result in state.get("verifier_results", []):
            bank_txn_id = candidate_verdict.matched_bank_txn_id
            if verifier_result.match and verifier_result.confidence >= 0.85:
                if bank_txn_id in matched_bank_ids:
                    continue
                matches.append(
                    MatchVerdict(
                        status=MatchStatus.MATCH,
                        match=True,
                        matched_settlement_id=verifier_result.matched_settlement_id,
                        matched_ledger_id=verifier_result.matched_ledger_id,
                        matched_bank_txn_id=bank_txn_id,
                        confidence=verifier_result.confidence,
                        pattern_detected=verifier_result.pattern_detected,
                        reasoning=verifier_result.reasoning,
                        evidence_fields_used=verifier_result.evidence_fields_used,
                        requires_verifier=False,
                    )
                )
                matched_bank_ids.add(bank_txn_id)
                continue

            exceptions.append(
                ExceptionRecord(
                    exception_id=f"EXC-{bank_txn_id or candidate_verdict.matched_settlement_id}",
                    bank_txn_id=bank_txn_id,
                    settlement_id=candidate_verdict.matched_settlement_id,
                    ledger_id=candidate_verdict.matched_ledger_id,
                    root_cause=verifier_result.pattern_detected,
                    severity=self._severity_for(verifier_result.pattern_detected),
                    confidence=verifier_result.confidence,
                    reason=verifier_result.reasoning,
                    evidence={
                        "candidate_pattern": candidate_verdict.pattern_detected.value,
                        "llm_match": verifier_result.match,
                    },
                )
            )

        return state | {"matches": matches, "exceptions": exceptions}

    def aggregate(self, state: ReconciliationState) -> ReconciliationState:
        summary = ReconciliationSummary(
            total_gateway_rows=len(state["gateway_rows"]),
            total_bank_rows=len(state["bank_rows"]),
            total_ledger_rows=len(state["ledger_rows"]),
            deterministic_matches=sum(
                1
                for match in state.get("matches", [])
                if match.pattern_detected == PatternDetected.CLEAN_MATCH
            ),
            verifier_candidates=len(state.get("verifier_queue", [])),
            exceptions=len(state.get("exceptions", [])),
        )
        return state | {"summary": summary}

    @staticmethod
    def _candidate_pool(
        gateway_rows: pd.DataFrame,
        ledger_rows: pd.DataFrame,
        *,
        excluded_settlement_ids: set[str | None],
    ) -> list[dict[str, Any]]:
        gateway = gateway_rows.copy()
        ledger = ledger_rows.copy()
        chains = gateway.merge(ledger, on="order_id", how="inner", validate="one_to_one")
        chains = chains[~chains["settlement_id"].isin(excluded_settlement_ids)]
        return [_jsonable(row) for row in chains.to_dict("records")]

    @staticmethod
    def _unmatched_bank_rows(
        bank_rows: pd.DataFrame,
        *,
        excluded_bank_txn_ids: set[str | None],
    ) -> list[dict[str, Any]]:
        rows = bank_rows[~bank_rows["txn_id"].isin(excluded_bank_txn_ids)]
        return [_jsonable(row) for row in rows.to_dict("records")]

    @staticmethod
    def _prioritise_declared_candidate(
        candidate_verdict: MatchVerdict,
        candidate_pool: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        declared = [
            candidate
            for candidate in candidate_pool
            if candidate.get("settlement_id") == candidate_verdict.matched_settlement_id
        ]
        rest = [
            candidate
            for candidate in candidate_pool
            if candidate.get("settlement_id") != candidate_verdict.matched_settlement_id
        ]
        return declared + rest

    @staticmethod
    def _severity_for(pattern: PatternDetected) -> ExceptionSeverity:
        if pattern == PatternDetected.AMOUNT_MISMATCH_WITH_INJECTION_ATTEMPT:
            return ExceptionSeverity.CRITICAL
        if pattern in {PatternDetected.DUPLICATE_BANK_ENTRY, PatternDetected.PARTIAL_REFUND}:
            return ExceptionSeverity.HIGH
        return ExceptionSeverity.MEDIUM


def build_reconciliation_graph(verifier: LLMVerifier | None = None) -> Any:
    return ReconciliationGraphBuilder(verifier=verifier).build()


def run_reconciliation(
    gateway_rows: pd.DataFrame,
    bank_rows: pd.DataFrame,
    ledger_rows: pd.DataFrame,
    verifier: LLMVerifier | None = None,
) -> ReconciliationState:
    graph = build_reconciliation_graph(verifier=verifier)
    return graph.invoke(
        {
            "gateway_rows": gateway_rows,
            "bank_rows": bank_rows,
            "ledger_rows": ledger_rows,
        }
    )


def run_reconciliation_from_csv(
    gateway_path: str | Path,
    bank_path: str | Path,
    ledger_path: str | Path,
    verifier: LLMVerifier | None = None,
) -> ReconciliationState:
    return run_reconciliation(
        pd.read_csv(gateway_path),
        pd.read_csv(bank_path),
        pd.read_csv(ledger_path),
        verifier=verifier,
    )


def _jsonable(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.isoformat() if hasattr(value, "isoformat") else value
        for key, value in row.items()
    }
