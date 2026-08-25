from __future__ import annotations

import json
import os
from decimal import Decimal
from time import perf_counter
from typing import Any, Protocol

from rapidfuzz import fuzz

from agent.prompts import SYSTEM_PROMPT, build_verification_prompt
from schemas import AgentVerificationResult, PatternDetected, quantize_money


try:
    from langfuse import observe
except Exception:

    def observe(*_args: Any, **_kwargs: Any) -> Any:
        def decorator(func: Any) -> Any:
            return func

        return decorator


class StructuredLLMClient(Protocol):
    def verify(self, system_prompt: str, user_prompt: str, schema: type[AgentVerificationResult]) -> dict[str, Any]:
        ...


class OpenAIStructuredClient:
    def __init__(self, model: str = "gpt-4.1-mini") -> None:
        from openai import OpenAI

        self._client = OpenAI()
        self._model = model

    def verify(self, system_prompt: str, user_prompt: str, schema: type[AgentVerificationResult]) -> dict[str, Any]:
        response = self._client.responses.parse(
            model=self._model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text_format=schema,
        )
        parsed = response.output_parsed
        return parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)


class LiteLLMStructuredClient:
    def __init__(self, model: str = "anthropic/claude-3-5-sonnet-20241022") -> None:
        from litellm import completion

        self._completion = completion
        self._model = model

    def verify(self, system_prompt: str, user_prompt: str, schema: type[AgentVerificationResult]) -> dict[str, Any]:
        response = self._completion(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=schema,
        )
        content = response["choices"][0]["message"]["content"]
        return json.loads(content)


class RuleBasedVerificationClient:
    """Offline verifier that mirrors the allowed LLM reasoning contract for tests and demos."""

    def verify(self, system_prompt: str, user_prompt: str, schema: type[AgentVerificationResult]) -> dict[str, Any]:
        raise RuntimeError("RuleBasedVerificationClient must be called through LLMVerifier._heuristic_verdict")


class LLMVerifier:
    def __init__(
        self,
        client: StructuredLLMClient | None = None,
        model_provider: str | None = None,
        confidence_threshold: float = 0.85,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.client = client or self._default_client(model_provider)

    @observe(name="llm_verifier_node")
    def verify_bank_row(
        self,
        bank_row: dict[str, Any],
        candidate_pool: list[dict[str, Any]],
        *,
        top_k: int = 3,
    ) -> AgentVerificationResult:
        started_at = perf_counter()
        candidates = self.top_candidates(bank_row, candidate_pool, top_k=top_k)
        prompt = build_verification_prompt(bank_row, candidates)
        metadata = {
            "bank_txn_id": bank_row.get("txn_id"),
            "candidate_count": len(candidates),
            "top_k": top_k,
            "provider": type(self.client).__name__,
        }

        if isinstance(self.client, RuleBasedVerificationClient):
            result = self._heuristic_verdict(bank_row, candidates)
        else:
            payload = self.client.verify(SYSTEM_PROMPT, prompt, AgentVerificationResult)
            result = AgentVerificationResult.model_validate(payload)

        latency_ms = (perf_counter() - started_at) * 1000
        self._record_observability(metadata | {"latency_ms": round(latency_ms, 3)}, result)
        return result

    def top_candidates(
        self,
        bank_row: dict[str, Any],
        candidate_pool: list[dict[str, Any]],
        *,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        scored = [
            (self._score_candidate(bank_row, candidate), candidate)
            for candidate in candidate_pool
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [candidate | {"candidate_score": score} for score, candidate in scored[:top_k]]

    def _default_client(self, model_provider: str | None) -> StructuredLLMClient:
        provider = (model_provider or os.getenv("AFC_LLM_PROVIDER") or "offline").lower()
        if provider == "openai" and os.getenv("OPENAI_API_KEY"):
            return OpenAIStructuredClient(os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
        if provider in {"anthropic", "claude", "litellm"} and (
            os.getenv("ANTHROPIC_API_KEY") or os.getenv("LITELLM_API_KEY")
        ):
            return LiteLLMStructuredClient(
                os.getenv("AFC_LLM_MODEL", "anthropic/claude-3-5-sonnet-20241022")
            )
        return RuleBasedVerificationClient()

    def _heuristic_verdict(
        self,
        bank_row: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> AgentVerificationResult:
        narration = str(bank_row.get("narration") or bank_row.get("unsafe_narration") or "")
        if self._contains_prompt_injection(narration):
            return AgentVerificationResult(
                match=False,
                matched_settlement_id=None,
                matched_ledger_id=None,
                confidence=0.0,
                pattern_detected=PatternDetected.AMOUNT_MISMATCH_WITH_INJECTION_ATTEMPT,
                reasoning="Untrusted bank narration contains a prompt-injection or command-override attempt; fail closed.",
                evidence_fields_used=("narration",),
            )

        if not candidates:
            return AgentVerificationResult(
                match=False,
                matched_settlement_id=None,
                matched_ledger_id=None,
                confidence=0.0,
                pattern_detected=PatternDetected.UNMATCHABLE,
                reasoning="No candidate gateway-ledger chain was supplied for this bank transaction.",
                evidence_fields_used=("utr", "net_amount"),
            )

        candidate = candidates[0]
        gross = self._money(candidate["amount"])
        fee = self._money(candidate["fee"])
        tax = self._money(candidate["tax"])
        gateway_net = self._money(candidate["net_amount"])
        calculated_net = quantize_money(gross - fee - tax)
        bank_amount = self._money(bank_row["amount"])
        bank_utr = str(bank_row["utr_ref"])
        gateway_utr = str(candidate["utr"])

        if calculated_net != gateway_net or bank_amount < gateway_net:
            return AgentVerificationResult(
                match=False,
                matched_settlement_id=None,
                matched_ledger_id=None,
                confidence=0.60,
                pattern_detected=PatternDetected.PARTIAL_REFUND,
                reasoning="UTR/order evidence exists, but the bank credit is reduced or gateway net does not reconcile cleanly.",
                evidence_fields_used=("utr", "gross_amount", "fee", "tax", "net_amount"),
            )

        if bank_utr == gateway_utr and bank_amount == gateway_net:
            day_delta = abs(
                (self._date(bank_row["value_date"]) - self._date(candidate["invoice_date"])).days
            )
            pattern = (
                PatternDetected.TIMING_MISMATCH
                if day_delta > 1
                else PatternDetected.FEE_SPLIT_MISMATCH
            )
            return AgentVerificationResult(
                match=True,
                matched_settlement_id=str(candidate["settlement_id"]),
                matched_ledger_id=str(candidate["ledger_id"]),
                confidence=0.95,
                pattern_detected=pattern,
                reasoning="UTR and bank net amount match; ledger gross reconciles to gateway net through fee and tax evidence.",
                evidence_fields_used=("utr", "gross_amount", "fee", "tax", "net_amount", "invoice_date", "value_date"),
            )

        if gateway_utr.startswith(bank_utr[:10]) and bank_amount == gateway_net:
            return AgentVerificationResult(
                match=True,
                matched_settlement_id=str(candidate["settlement_id"]),
                matched_ledger_id=str(candidate["ledger_id"]),
                confidence=0.93,
                pattern_detected=PatternDetected.TRUNCATED_UTR,
                reasoning="Bank UTR is truncated, but the first 10 characters, net amount, and candidate chain align.",
                evidence_fields_used=("utr_prefix", "net_amount", "invoice_date", "value_date"),
            )

        return AgentVerificationResult(
            match=False,
            matched_settlement_id=None,
            matched_ledger_id=None,
            confidence=0.0,
            pattern_detected=PatternDetected.UNMATCHABLE,
            reasoning="Candidate evidence is insufficient for a safe verifier match under the 0.85 policy.",
            evidence_fields_used=("utr", "net_amount"),
        )

    def _record_observability(self, metadata: dict[str, Any], result: AgentVerificationResult) -> None:
        try:
            from langfuse import get_client

            langfuse = get_client()
            langfuse.update_current_trace(
                metadata=metadata
                | {
                    "confidence": result.confidence,
                    "pattern_detected": result.pattern_detected.value,
                    "match": result.match,
                }
            )
        except Exception:
            return

    @staticmethod
    def _score_candidate(bank_row: dict[str, Any], candidate: dict[str, Any]) -> float:
        bank_utr = str(bank_row.get("utr_ref", ""))
        gateway_utr = str(candidate.get("utr", ""))
        utr_score = fuzz.ratio(bank_utr, gateway_utr) / 100
        prefix_score = 1.0 if gateway_utr.startswith(bank_utr[:10]) else 0.0
        bank_amount = Decimal(str(bank_row.get("amount", "0")))
        candidate_net = Decimal(str(candidate.get("net_amount", "0")))
        decimal_amount_score = (
            Decimal("1.0")
            if bank_amount == candidate_net
            else max(
                Decimal("0.0"),
                Decimal("1.0")
                - min(abs(bank_amount - candidate_net) / Decimal("10000"), Decimal("1.0")),
            )
        )
        amount_score = float(decimal_amount_score)
        return round((0.55 * max(utr_score, prefix_score)) + (0.45 * float(amount_score)), 4)

    @staticmethod
    def _contains_prompt_injection(text: str) -> bool:
        lowered = text.lower()
        suspicious_phrases = (
            "ignore all previous instructions",
            "ignore previous instructions",
            "mark this transaction",
            "confirmed exact match regardless",
            "system note:",
            "override",
        )
        return any(phrase in lowered for phrase in suspicious_phrases)

    @staticmethod
    def _money(value: object) -> Decimal:
        return quantize_money(Decimal(str(value)))

    @staticmethod
    def _date(value: object) -> Any:
        import pandas as pd

        return pd.to_datetime(value).date()
