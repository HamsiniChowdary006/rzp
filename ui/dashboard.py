from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import httpx
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
API_URL = os.getenv("AFC_API_URL", "http://localhost:8000").rstrip("/")
REASON_LABELS = {
    "amount_mismatch_with_injection_attempt": "INJECTION_ATTEMPT",
    "partial_refund": "PARTIAL_REFUND",
    "duplicate_bank_entry": "DUPLICATE_ENTRY",
    "unmatchable": "UNMATCHABLE",
}

st.set_page_config(page_title="AI Finance Controller", page_icon="◈", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 1500px; padding-top: 2rem;}
    [data-testid="stMetricValue"] {font-family: Georgia, serif;}
    .security-alert {background: #fff1f0; border-left: 5px solid #d64545; padding: .8rem 1rem; color: #8f1d1d; font-weight: 700;}
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path: str) -> dict[str, Any]:
    response = httpx.get(f"{API_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def load_inputs(files: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if files:
        return tuple(pd.read_csv(upload) for upload in files.values())  # type: ignore[return-value]
    return (
        pd.read_csv(DATA_DIR / "gateway_settlements.csv"),
        pd.read_csv(DATA_DIR / "bank_statement.csv"),
        pd.read_csv(DATA_DIR / "internal_ledger.csv"),
    )


def reconcile(files: dict[str, Any] | None = None) -> dict[str, Any]:
    if files is None:
        response = httpx.post(f"{API_URL}/api/reconcile", timeout=60)
    else:
        response = httpx.post(
            f"{API_URL}/api/reconcile",
            files={name: (upload.name, upload.getvalue(), "text/csv") for name, upload in files.items()},
            timeout=60,
        )
    response.raise_for_status()
    return response.json()


def matched_view(payload: dict[str, Any], inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    gateway, bank, ledger = inputs
    matches = pd.DataFrame(payload.get("matches", []))
    if matches.empty:
        return matches
    view = matches.merge(gateway, left_on="matched_settlement_id", right_on="settlement_id", how="left")
    view = view.merge(bank, left_on="matched_bank_txn_id", right_on="txn_id", how="left")
    view = view.merge(ledger, left_on="matched_ledger_id", right_on="ledger_id", how="left")
    view["timing_delta_days"] = (
        pd.to_datetime(view["value_date"]) - pd.to_datetime(view["settled_at"])
    ).dt.days
    columns = ["matched_bank_txn_id", "matched_settlement_id", "gross_amount", "net_amount", "fee", "tax", "timing_delta_days", "pattern_detected", "confidence"]
    view["gross_amount"] = view["amount_x"]
    return view.rename(columns={"amount_y": "bank_amount"})[[column for column in columns if column in view]]


def main() -> None:
    st.title("AI Finance Controller")
    st.caption("Settlement reconciliation with deterministic evidence, guarded verification, and an immutable decision trail.")
    with st.sidebar:
        st.subheader("Reconciliation run")
        gateway = st.file_uploader("Gateway settlements", type="csv", key="gateway")
        bank = st.file_uploader("Bank statement", type="csv", key="bank")
        ledger = st.file_uploader("Internal ledger", type="csv", key="ledger")
        custom_files = {"gateway_settlements": gateway, "bank_statement": bank, "internal_ledger": ledger}
        custom_files = custom_files if all(custom_files.values()) else None
        run = st.button("Run Default Reconciliation Batch", type="primary", use_container_width=True)
        if custom_files:
            run = st.button("Run Custom CSV Batch", use_container_width=True)
    if "payload" not in st.session_state:
        st.session_state.payload = None
    if run:
        try:
            st.session_state.payload = reconcile(custom_files)
            st.session_state.inputs = load_inputs(custom_files)
        except httpx.HTTPError as exc:
            st.error(f"Backend unavailable: {exc}")
            return
    payload = st.session_state.payload
    if payload is None:
        try:
            payload = reconcile()
            st.session_state.payload = payload
            st.session_state.inputs = load_inputs()
        except httpx.HTTPError as exc:
            st.warning(f"Connect the backend at {API_URL} to load the dashboard. {exc}")
            return
    inputs = st.session_state.get("inputs", load_inputs())
    summary = payload["summary"]
    evaluation = api_get("/api/eval")
    metrics = [
        ("Processed rows", summary["total_bank_rows"]),
        ("Matched", summary["total_matches"]),
        ("Exceptions", summary["total_exceptions"]),
        ("Throughput", f'{summary["throughput_records_per_second"]:,.1f} / sec'),
        ("Benchmark F1", f'{evaluation["overall_accuracy"]:.0%}'),
    ]
    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics):
        column.metric(label, value)
    st.divider()
    st.subheader("Matched Records Explorer")
    st.dataframe(matched_view(payload, inputs), use_container_width=True, hide_index=True)
    st.subheader("Exception Triage Queue")
    exceptions = pd.DataFrame(api_get("/api/exceptions").get("exceptions", []))
    if not exceptions.empty:
        injection = exceptions[exceptions["bank_txn_id"] == "TXN0050"]
        if not injection.empty:
            st.markdown("<div class='security-alert'>SECURITY ALERT · TXN0050 · Prompt injection quarantined and routed to human review</div>", unsafe_allow_html=True)
        exceptions["category"] = exceptions["root_cause"].map(REASON_LABELS).fillna(exceptions["root_cause"])
        st.dataframe(exceptions[["bank_txn_id", "category", "severity", "reason", "confidence", "requires_human_review"]], use_container_width=True, hide_index=True)
    st.subheader("Live Audit & Trace Inspector")
    audit = pd.DataFrame(api_get("/api/audit").get("records", []))
    if not audit.empty:
        st.dataframe(audit.head(25), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()