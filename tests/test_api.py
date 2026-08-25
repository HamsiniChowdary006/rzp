from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from api.app import app


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_reconcile_default_dataset_returns_expected_counts() -> None:
    with TestClient(app) as client:
        response = client.post("/api/reconcile")

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["total_matches"] == 47
    assert payload["summary"]["total_exceptions"] == 6


def test_reconcile_file_upload_mode_returns_expected_counts() -> None:
    with (
        (DATA_DIR / "gateway_settlements.csv").open("rb") as gateway,
        (DATA_DIR / "bank_statement.csv").open("rb") as bank,
        (DATA_DIR / "internal_ledger.csv").open("rb") as ledger,
        TestClient(app) as client,
    ):
        response = client.post(
            "/api/reconcile",
            files={
                "gateway_settlements": ("gateway_settlements.csv", gateway, "text/csv"),
                "bank_statement": ("bank_statement.csv", bank, "text/csv"),
                "internal_ledger": ("internal_ledger.csv", ledger, "text/csv"),
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset"] == "uploaded"
    assert payload["summary"]["total_matches"] == 47
    assert payload["summary"]["total_exceptions"] == 6


def test_eval_returns_perfect_benchmark_metrics() -> None:
    with TestClient(app) as client:
        response = client.get("/api/eval")

    assert response.status_code == 200
    payload = response.json()
    assert payload["matches"]["precision"] == 1.0
    assert payload["matches"]["recall"] == 1.0
    assert payload["matches"]["f1_score"] == 1.0
    assert payload["exceptions"]["precision"] == 1.0
    assert payload["exceptions"]["recall"] == 1.0
    assert payload["exceptions"]["f1_score"] == 1.0
    assert payload["overall_accuracy"] == 1.0


def test_exceptions_contains_txn0050_prompt_injection() -> None:
    with TestClient(app) as client:
        response = client.get("/api/exceptions")

    assert response.status_code == 200
    payload = response.json()
    injection = [
        exception
        for exception in payload["exceptions"]
        if exception["bank_txn_id"] == "TXN0050"
    ]
    assert len(injection) == 1
    assert injection[0]["reason_code"] == "INJECTION_ATTEMPT"
    assert injection[0]["root_cause"] == "amount_mismatch_with_injection_attempt"


def test_health_returns_service_status() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "version" in payload
    assert "active_llm_provider" in payload


def test_audit_returns_non_empty_structured_decision_logs() -> None:
    with TestClient(app) as client:
        client.post("/api/reconcile")
        response = client.get("/api/audit")

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] > 0
    first_record = payload["records"][0]
    assert {"timestamp", "run_id", "event_type", "record_type", "payload"} <= set(first_record)
