# AI Finance Controller

AI Finance Controller is a deterministic-first reconciliation service for Razorpay-style gateway settlements, bank statements, and an internal ledger. It processes the supplied 53-row batch, explains every decision, and escalates ambiguity instead of silently inventing a match.

## Razorpay Evaluation Criteria

### 1. Problem Taste

Reconciliation friction lives at the boundaries between three systems: settlement files describe gross money and deductions, bank feeds show credits on their own dates, and the ledger carries the merchant's expected amount. Manual teams spend time joining identifiers, explaining fee splits, and investigating duplicate or incomplete credits. In fintech, verification is more valuable than unconstrained generation: a wrong auto-match can post money to the wrong order, while a transparent exception is recoverable.

### 2. System Architecture

```text
CSV uploads / default batch
          |
          v
Pydantic validation -> Deterministic Matcher (UTR, amounts, dates, fuzzy prefixes)
          | clean evidence                 | ambiguous candidates
          v                                v
     Match result -> LangGraph Verifier (bounded, structured tool output)
          |                                |
          +-------------> Decision Gate <---+
                              |
                 +------------+-------------+
                 v                          v
           Match explorer             Human exception queue
                              |
                              v
                    Immutable append-only JSONL audit log
```

FastAPI exposes `/api/reconcile`, `/api/eval`, `/api/exceptions`, and `/api/audit`. Streamlit consumes those contracts and enriches match IDs with the source-file financial fields for operations review.

### 3. AI Judgment and Security

The deterministic pass handles exact UTR and amount relationships first. Only ambiguous rows reach the LangGraph verifier, which receives one bank row and at most three candidates. Pydantic models reject extra fields and constrain IDs, confidence, amounts, and allowed evidence fields. The verifier can attest to a decision but cannot mutate accounting state.

Bank narration and ledger customer values are untrusted data, quarantined inside explicit XML data delimiters before prompt construction. `TXN0050` contains an injection attempt and an amount mismatch; the decision gate keeps it out of the match set, marks it critical, and requires human review. Secrets belong in `.env`, which is ignored; `.env.example` contains placeholders only.

### 4. Failure Recovery Deep-Dive

| Failure | Detection and recovery |
| --- | --- |
| Prompt injection (`TXN0050`) | The malicious narration is treated as data, not instructions. Amount evidence fails, the row is classified as `INJECTION_ATTEMPT`, and the critical exception is quarantined for a human. |
| Partial refund (`TXN0049`) | UTR linkage is insufficient to approve a reduced credit. The gross/net discrepancy becomes `PARTIAL_REFUND`, preserving the evidence for a refund-ledger cross-check. |
| Bank deduplication (`TXN0048`) | Once a UTR has been consumed by a valid match, a repeated bank credit cannot reuse that settlement. It is classified as `DUPLICATE_ENTRY` and escalated rather than double-counted. |

## Quickstart

Use Python 3.13+ in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
pytest
```

Run the services locally in separate terminals:

```powershell
uvicorn api.app:app --reload --port 8000
streamlit run ui/dashboard.py
```

Open `http://localhost:8501`. The API is available at `http://localhost:8000/docs`.

For the complete stack:

```powershell
Copy-Item .env.example .env
docker-compose up --build
```

The dashboard is on port 8501 and the backend is on port 8000. Audit data is stored in a Docker volume and is not committed to the repository.

## Benchmark

The default dataset contains 47 matches and 6 exceptions across 53 bank rows. The test suite validates 100% precision, recall, and F1 for the supplied ground truth, including clean matches, fee splits, timing differences, truncated UTRs, partial refunds, duplicates, unmatchable credits, and the injection attempt.

## Repository Layout

```text
api/          FastAPI application and endpoints
agent/        LangGraph workflow and verifier boundary
matcher/      Deterministic reconciliation engine
ui/           Streamlit operations dashboard
data/         Synthetic evaluation inputs
tests/        Unit and API tests
```
