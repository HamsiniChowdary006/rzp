from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any, Annotated
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent.graph import run_reconciliation
from agent.verifier import LLMVerifier, RuleBasedVerificationClient
from evaluator import DEFAULT_DATA_DIR, EvaluationReport, evaluate_default_dataset
from schemas import ExceptionRecord, MatchVerdict


APP_VERSION = "0.3.0"
AUDIT_LOG_PATH = Path(os.getenv("AFC_AUDIT_LOG_PATH", DEFAULT_DATA_DIR / "audit_log.jsonl"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    app.state.verifier = LLMVerifier()
    yield


def create_app() -> FastAPI:
    application = FastAPI(
        title="AI Finance Controller",
        version=APP_VERSION,
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("AFC_CORS_ORIGINS", "*").split(","),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.add_exception_handler(RequestValidationError, _validation_exception_handler)
    application.add_exception_handler(Exception, _unhandled_exception_handler)

    @application.get("/health")
    def health() -> dict[str, str]:
        verifier = getattr(application.state, "verifier", LLMVerifier())
        return {
            "status": "ok",
            "version": APP_VERSION,
            "active_llm_provider": type(verifier.client).__name__,
        }

    @application.post("/api/reconcile")
    async def reconcile_endpoint(
        gateway_file: Annotated[UploadFile | None, File(alias="gateway_settlements")] = None,
        bank_file: Annotated[UploadFile | None, File(alias="bank_statement")] = None,
        ledger_file: Annotated[UploadFile | None, File(alias="internal_ledger")] = None,
    ) -> dict[str, Any]:
        started_at = perf_counter()
        gateway_rows, bank_rows, ledger_rows, dataset_name = await _load_reconciliation_inputs(
            gateway_file,
            bank_file,
            ledger_file,
        )
        state = run_reconciliation(
            gateway_rows,
            bank_rows,
            ledger_rows,
            verifier=getattr(application.state, "verifier", None),
        )
        elapsed_ms = (perf_counter() - started_at) * 1000
        response = _reconciliation_response(state, elapsed_ms=elapsed_ms, dataset_name=dataset_name)
        _append_audit_run(
            event_type="reconcile",
            run_id=str(uuid4()),
            payload=response,
        )
        return response

    @application.get("/api/eval", response_model=EvaluationReport)
    def eval_endpoint() -> EvaluationReport:
        report = evaluate_default_dataset()
        _append_audit_run(
            event_type="eval",
            run_id=str(uuid4()),
            payload=report.model_dump(mode="json"),
        )
        return report

    @application.get("/api/exceptions")
    def exceptions_endpoint() -> dict[str, Any]:
        state = _run_default_state(application)
        exceptions = [_exception_to_triage(exception) for exception in state["exceptions"]]
        return {"count": len(exceptions), "exceptions": exceptions}

    @application.get("/api/audit")
    def audit_endpoint() -> dict[str, Any]:
        records = _read_audit_records()
        if not records:
            state = _run_default_state(application)
            response = _reconciliation_response(state, elapsed_ms=0.0, dataset_name="default")
            _append_audit_run(event_type="audit_seed", run_id=str(uuid4()), payload=response)
            records = _read_audit_records()
        return {"count": len(records), "records": records}

    return application


async def _validation_exception_handler(_request: Any, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"error": "validation_error", "details": jsonable_encoder(exc.errors())},
    )


async def _unhandled_exception_handler(_request: Any, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "detail": str(exc)},
    )


async def _load_reconciliation_inputs(
    gateway_file: UploadFile | None,
    bank_file: UploadFile | None,
    ledger_file: UploadFile | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    files = [gateway_file, bank_file, ledger_file]
    if all(file is None for file in files):
        return (
            pd.read_csv(DEFAULT_DATA_DIR / "gateway_settlements.csv"),
            pd.read_csv(DEFAULT_DATA_DIR / "bank_statement.csv"),
            pd.read_csv(DEFAULT_DATA_DIR / "internal_ledger.csv"),
            "default",
        )
    if any(file is None for file in files):
        raise HTTPException(
            status_code=400,
            detail="upload all three CSV files or omit all files to use the default dataset",
        )
    assert gateway_file is not None and bank_file is not None and ledger_file is not None
    return (
        await _read_upload_csv(gateway_file),
        await _read_upload_csv(bank_file),
        await _read_upload_csv(ledger_file),
        "uploaded",
    )


async def _read_upload_csv(file: UploadFile) -> pd.DataFrame:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail=f"{file.filename or 'upload'} must be a CSV file")
    content = await file.read()
    try:
        return pd.read_csv(BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid CSV upload {file.filename}: {exc}") from exc


def _run_default_state(application: FastAPI) -> dict[str, Any]:
    return run_reconciliation(
        pd.read_csv(DEFAULT_DATA_DIR / "gateway_settlements.csv"),
        pd.read_csv(DEFAULT_DATA_DIR / "bank_statement.csv"),
        pd.read_csv(DEFAULT_DATA_DIR / "internal_ledger.csv"),
        verifier=getattr(application.state, "verifier", None),
    )


def _reconciliation_response(
    state: dict[str, Any],
    *,
    elapsed_ms: float,
    dataset_name: str,
) -> dict[str, Any]:
    matches = [match.model_dump(mode="json") for match in state.get("matches", [])]
    exceptions = [exception.model_dump(mode="json") for exception in state.get("exceptions", [])]
    records_processed = len(state["bank_rows"])
    return {
        "dataset": dataset_name,
        "summary": {
            "total_matches": len(matches),
            "total_exceptions": len(exceptions),
            "total_bank_rows": records_processed,
            "latency_ms": round(elapsed_ms, 3),
            "throughput_records_per_second": round(records_processed / (elapsed_ms / 1000), 3)
            if elapsed_ms
            else 0.0,
        },
        "matches": matches,
        "exceptions": exceptions,
    }


def _exception_to_triage(exception: ExceptionRecord) -> dict[str, Any]:
    return {
        "exception_id": exception.exception_id,
        "bank_txn_id": exception.bank_txn_id,
        "settlement_id": exception.settlement_id,
        "ledger_id": exception.ledger_id,
        "reason_code": _reason_code(exception),
        "root_cause": exception.root_cause.value,
        "severity": exception.severity.value,
        "confidence": exception.confidence,
        "reason": exception.reason,
        "requires_human_review": exception.requires_human_review,
    }


def _reason_code(exception: ExceptionRecord) -> str:
    mapping = {
        "amount_mismatch_with_injection_attempt": "INJECTION_ATTEMPT",
        "partial_refund": "PARTIAL_REFUND",
        "duplicate_bank_entry": "DUPLICATE_ENTRY",
        "unmatchable": "UNMATCHABLE",
    }
    return mapping.get(exception.root_cause.value, exception.root_cause.value.upper())


def _append_audit_run(*, event_type: str, run_id: str, payload: dict[str, Any]) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, Any]] = [
        {
            "timestamp": timestamp,
            "run_id": run_id,
            "event_type": event_type,
            "record_type": "run_summary",
            "langfuse_trace_url": os.getenv("LANGFUSE_HOST"),
            "payload": payload.get("summary", payload),
        }
    ]
    # Only iterate over matches/exceptions if they are lists (reconcile payloads).
    # For EvaluationReport payloads, matches/exceptions are ClassificationMetrics dicts.
    matches_data = payload.get("matches", [])
    if isinstance(matches_data, list):
        for match in matches_data:
            if isinstance(match, dict):
                records.append(_decision_audit_record(timestamp, run_id, event_type, "match", match))
    exceptions_data = payload.get("exceptions", [])
    if isinstance(exceptions_data, list):
        for exception in exceptions_data:
            if isinstance(exception, dict):
                records.append(_decision_audit_record(timestamp, run_id, event_type, "exception", exception))

    with AUDIT_LOG_PATH.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(jsonable_encoder(record), sort_keys=True) + "\n")


def _decision_audit_record(
    timestamp: str,
    run_id: str,
    event_type: str,
    record_type: str,
    decision: dict[str, Any],
) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "run_id": run_id,
        "event_type": event_type,
        "record_type": record_type,
        "bank_txn_id": decision.get("matched_bank_txn_id") or decision.get("bank_txn_id"),
        "settlement_id": decision.get("matched_settlement_id") or decision.get("settlement_id"),
        "ledger_id": decision.get("matched_ledger_id") or decision.get("ledger_id"),
        "pattern": decision.get("pattern_detected") or decision.get("root_cause"),
        "confidence": decision.get("confidence"),
        "langfuse_trace_url": os.getenv("LANGFUSE_HOST"),
        "decision": decision,
    }


def _read_audit_records() -> list[dict[str, Any]]:
    if not AUDIT_LOG_PATH.exists():
        return []
    records: list[dict[str, Any]] = []
    with AUDIT_LOG_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


app = create_app()
