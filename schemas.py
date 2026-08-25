from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Money = Annotated[Decimal, Field(ge=Decimal("0.00"), max_digits=14, decimal_places=2)]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )


def quantize_money(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


class PatternDetected(str, Enum):
    CLEAN_MATCH = "clean_match"
    FEE_SPLIT_MISMATCH = "fee_split_mismatch"
    TIMING_MISMATCH = "timing_mismatch"
    TRUNCATED_UTR = "truncated_utr"
    DUPLICATE_BANK_ENTRY = "duplicate_bank_entry"
    PARTIAL_REFUND = "partial_refund"
    AMOUNT_MISMATCH_WITH_INJECTION_ATTEMPT = "amount_mismatch_with_injection_attempt"
    UNMATCHABLE = "unmatchable"
    UNKNOWN = "unknown"


class MatchStatus(str, Enum):
    MATCH = "match"
    EXCEPTION = "exception"
    NEEDS_VERIFIER = "needs_verifier"


class ExceptionSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class GatewaySettlementRow(StrictBaseModel):
    settlement_id: str = Field(pattern=r"^STL\d{4}$")
    utr: str = Field(min_length=10, max_length=32, pattern=r"^[A-Z0-9]+$")
    amount: Money
    fee: Money
    tax: Money
    net_amount: Money
    settled_at: date
    order_id: str = Field(pattern=r"^ORD\d{4}$")

    @field_validator("amount", "fee", "tax", "net_amount", mode="before")
    @classmethod
    def _money(cls, value: object) -> Decimal:
        return quantize_money(Decimal(str(value)))

    @field_validator("utr")
    @classmethod
    def _normalise_utr(cls, value: str) -> str:
        return value.strip().upper()

class BankStatementRow(StrictBaseModel):
    txn_id: str = Field(pattern=r"^TXN\d{4}$")
    utr_ref: str = Field(min_length=4, max_length=32, pattern=r"^[A-Z0-9]+$")
    amount: Money
    value_date: date
    narration: str = Field(min_length=1, max_length=512)

    @field_validator("amount", mode="before")
    @classmethod
    def _money(cls, value: object) -> Decimal:
        return quantize_money(Decimal(str(value)))

    @field_validator("utr_ref")
    @classmethod
    def _normalise_utr_ref(cls, value: str) -> str:
        return value.strip().upper()


class InternalLedgerRow(StrictBaseModel):
    ledger_id: str = Field(pattern=r"^LED\d{4}$")
    order_id: str = Field(pattern=r"^ORD\d{4}$")
    expected_amount: Money
    invoice_date: date
    customer: str = Field(min_length=1, max_length=256)

    @field_validator("expected_amount", mode="before")
    @classmethod
    def _money(cls, value: object) -> Decimal:
        return quantize_money(Decimal(str(value)))


class MatchVerdict(StrictBaseModel):
    status: MatchStatus = MatchStatus.MATCH
    match: bool
    matched_settlement_id: str | None = Field(default=None, pattern=r"^STL\d{4}$")
    matched_ledger_id: str | None = Field(default=None, pattern=r"^LED\d{4}$")
    matched_bank_txn_id: str | None = Field(default=None, pattern=r"^TXN\d{4}$")
    confidence: Confidence
    pattern_detected: PatternDetected
    reasoning: str = Field(min_length=1, max_length=1000)
    evidence_fields_used: tuple[str, ...] = Field(default_factory=tuple, min_length=1)
    requires_verifier: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _matched_ids_required_for_matches(self) -> "MatchVerdict":
        if self.match and not (
            self.matched_settlement_id and self.matched_ledger_id and self.matched_bank_txn_id
        ):
            raise ValueError("matched settlement, ledger, and bank IDs are required for matches")
        if self.confidence < 0.85 and not self.requires_verifier and self.status == MatchStatus.MATCH:
            raise ValueError("low-confidence matches must be routed to verifier or exception queue")
        return self


class ExceptionRecord(StrictBaseModel):
    exception_id: str
    bank_txn_id: str | None = Field(default=None, pattern=r"^TXN\d{4}$")
    settlement_id: str | None = Field(default=None, pattern=r"^STL\d{4}$")
    ledger_id: str | None = Field(default=None, pattern=r"^LED\d{4}$")
    root_cause: PatternDetected
    severity: ExceptionSeverity = ExceptionSeverity.MEDIUM
    confidence: Confidence = 0.0
    reason: str = Field(min_length=1, max_length=1200)
    evidence: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    requires_human_review: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ReconciliationSummary(StrictBaseModel):
    total_gateway_rows: int = Field(ge=0)
    total_bank_rows: int = Field(ge=0)
    total_ledger_rows: int = Field(ge=0)
    deterministic_matches: int = Field(ge=0)
    verifier_candidates: int = Field(ge=0)
    exceptions: int = Field(ge=0)
    precision_against_ground_truth: Confidence | None = None
    recall_against_ground_truth: Confidence | None = None
    f1_against_ground_truth: Confidence | None = None
    elapsed_ms: float | None = Field(default=None, ge=0.0)


class AgentVerificationResult(StrictBaseModel):
    """Least-privilege LLM verdict. The agent can only attest; it cannot mutate state."""

    match: bool
    matched_settlement_id: str | None = Field(default=None, pattern=r"^STL\d{4}$")
    matched_ledger_id: str | None = Field(default=None, pattern=r"^LED\d{4}$")
    confidence: Confidence
    pattern_detected: PatternDetected
    reasoning: str = Field(min_length=20, max_length=1000)
    evidence_fields_used: tuple[
        Literal[
            "utr",
            "utr_prefix",
            "gross_amount",
            "net_amount",
            "fee",
            "tax",
            "order_id",
            "settled_at",
            "value_date",
            "invoice_date",
            "narration",
            "customer",
        ],
        ...,
    ] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def _consistent_match_payload(self) -> "AgentVerificationResult":
        if self.match and not (self.matched_settlement_id and self.matched_ledger_id):
            raise ValueError("matched IDs are required when match=true")
        if not self.match and self.confidence > 0.85:
            raise ValueError("non-match confidence must not imply auto-postable certainty")
        return self
