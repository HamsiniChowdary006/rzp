Here is the complete, self-contained `SPEC.md` file updated to include all requested technologies (FastAPI, LangGraph, Langfuse, Pydantic, RapidFuzz), detailed edge-case patterns from your dataset, and concrete deliverables for each build phase.

```markdown
# AI Finance Controller — Execution Plan
### Track 04, Razorpay AI Buildathon 2026 — Multi-Source Reconciliation Agent

---

## 1. Product Definition (lock this before writing any code)

**One-line pitch:** An agent that reconciles a merchant's Payment Gateway settlement file, Bank statement, and internal Ledger, produces a matched/unmatched report with confidence-scored reasoning, and hands off a clean exception queue instead of silently guessing[cite: 1].

**Why this framing wins on the judging bar:**
- **Throughput:** Process 50+ records in batches, reporting execution latency and records-per-second[cite: 1].
- **Measured Accuracy:** Benchmark match rate, precision, recall, and F1-score against a known `ground_truth.csv` evaluation set[cite: 1].
- **Honest Exception List:** Anything below a confidence threshold goes to a human-review queue with an explicit root-cause classification, not a forced match[cite: 1].
- **Full Observability & Security:** Complete tracing of all LLM calls, latency, token spend, and proactive defense against prompt injection in raw accounting feeds[cite: 1].

**Non-goals (say these explicitly in your pitch — scope discipline is a signal):**
- Not building a general ledger or complete accounting system[cite: 1].
- Not integrating live banking APIs (operates on versioned synthetic batch datasets)[cite: 1].
- Not trying to auto-resolve 100% — forcing every match is a red flag, not a feature[cite: 1].

---

## 2. Architecture


```

┌─────────────────────────────────────────────────────────────┐
│  Data Ingestion Layer (Pydantic validated)                   │
│  - gateway_settlements.csv  (Razorpay-style settlement file) │
│  - bank_statement.csv                                       │
│  - internal_ledger.csv                                       │
└───────────────┬─────────────────────────────────────────────┘
│
┌───────▼────────────────────────┐
│  Deterministic Match Engine    │  Rule-based pass (pandas + rapidfuzz):
│  (FastAPI / Python Node)       │  - Exact UTR + Net Amount match
│                                │  - Gross vs Net fee calculation
│                                │  - Truncated UTR prefix match (10 chars)
└───────┬────────────────────────┘
│
├───────────────────────────────────────┐
│ Clean matches (33 rows)               │ Ambiguous rows (20 rows)
▼                                       ▼
┌────────────────┐              ┌───────────────────────────────┐
│ Matched Record │              │ LangGraph LLM Verifier Node   │
│ Store          │              │ (Claude 3.5 Sonnet Tool Call) │
└────────────────┘              │ - Traced with Langfuse        │
│ - Untrusted input quarantine  │
└───────────────┬───────────────┘
│
┌───────▼────────┐
│ Decision Gate  │
└───────┬────────┘
│
├───────────────────────┴───────────────────────┐
│ Confidence ≥ 0.85                             │ Confidence < 0.85 / Anomaly
▼                                               ▼
┌────────────────┐                              ┌────────────────┐
│ Matched Record │                              │ Exception      │
│ Store          │                              │ Queue + Reason │
└────────────────┘                              └────────────────┘

```

**Key design decision:** The LLM is a *verifier for the hard 10–20%*, not the primary matcher[cite: 1]. This matches Razorpay's stated thesis ("verification capacity, not generation speed, is the bottleneck")[cite: 1]. It also gives you a much stronger security, latency, and cost story than throwing every row at an LLM[cite: 1].

---

## 3. Tech Stack

| Layer | Choice | Justification |
|---|---|---|
| **Backend API** | FastAPI (Python 3.11+) | Asynchronous, typed endpoints for `/reconcile`, `/eval`, and `/audit`[cite: 1]. |
| **Data Validation** | Pydantic v2 | Strict schema enforcement, sanitizing numeric amounts and timestamps[cite: 1]. |
| **Deterministic Matching** | pandas + rapidfuzz | Zero-cost, high-speed matching on exact and fuzzy numeric/UTR keys[cite: 1]. |
| **Agent Orchestration** | LangGraph | Stateful workflow routing between deterministic rules, candidate retrieval, and LLM verification. |
| **Agent Reasoning** | Claude 3.5 Sonnet / OpenAI | Tool-calling enforcing structured JSON verdicts and explicit rationales[cite: 1]. |
| **Observability** | Langfuse | Captures traces, prompt versions, execution latency, and cost per verification call. |
| **Storage & Audit** | SQLite + JSONL | Fast embedded persistence with append-only immutable audit logging[cite: 1]. |
| **Frontend UI** | Streamlit or Next.js | Interactive dashboard displaying match statistics, exception triage, and audit trail[cite: 1]. |
| **Packaging & CI** | Docker & `docker-compose` | One-command setup (`docker-compose up`) ensuring complete reproducibility[cite: 1]. |

---

## 4. Data Model & Test Dataset Patterns

### Input Schemas

* **`gateway_settlements.csv`:** `settlement_id`, `utr`, `amount`, `fee`, `tax`, `net_amount`, `settled_at`, `order_id`[cite: 1]
* **`bank_statement.csv`:** `txn_id`, `utr_ref`, `amount`, `value_date`, `narration`[cite: 1]
* **`internal_ledger.csv`:** `ledger_id`, `order_id`, `expected_amount`, `invoice_date`, `customer`[cite: 1]

### Test Edge-Case Breakdown (53 Records vs Ground Truth)

| Pattern | Count | Match Strategy & Resolution |
|---|---|---|
| `clean_match` | 33 | 3-way exact alignment across UTR, net amount, and standard T+1 date windows[cite: 1]. |
| `fee_split_mismatch` | 6 | Ledger gross matches Gateway gross; Bank net matches Gateway net (`amount - fee - tax`)[cite: 1]. |
| `timing_mismatch` | 5 | Value date landed outside standard T+1 window (T+2 or T+3), resolved via UTR and net amount alignment[cite: 1]. |
| `truncated_utr` | 3 | Bank narration truncates UTR to 10 characters; prefix + amount + date confirmation[cite: 1]. |
| `duplicate_bank_entry` | 1 | Duplicate bank credit for an already settled UTR; flagged to prevent double counting[cite: 1]. |
| `partial_refund` | 1 | Net credit reduced by refund amount; escalated to exception queue for refund-ledger cross-check[cite: 1]. |
| `amount_mismatch_with_injection_attempt` | 1 | Malicious prompt in narration attempting auto-match; must fail safely into exception queue[cite: 1]. |
| `unmatchable` | 3 | Extraneous credits with no ledger/gateway counterpart; escalated to finance ops[cite: 1]. |

---

## 5. Agent Design & Tool-Calling Specification

- **Structured Output Only:** Enforce JSON schema via tool-calling:
  ```json
  {
    "match": true,
    "matched_settlement_id": "STL0033",
    "matched_ledger_id": "LED0033",
    "confidence": 0.95,
    "pattern_detected": "fee_split_mismatch",
    "reasoning": "Ledger gross amount 7157.02 minus fee 178.93 and tax 32.21 equals bank net amount 6945.88. UTR chains match.",
    "evidence_fields_used": ["utr", "net_amount", "fee", "tax"]
  }

```

* **Bounded Context per Call:** Pass only the target bank row + top-3 candidate matches from the deterministic pass — never the whole dataset.


* **Untrusted Field Quarantine:** Field values from `narration` and `customer` are enclosed in explicit XML data delimiters (`<untrusted_narration>...</untrusted_narration>`) and ignored during instruction parsing.


* **Explicit "I Don't Know" Path:** System prompt rewards `confidence: low` over forced guesses. Any verdict with `confidence < 0.85` or `match == false` automatically routes to the human exception queue.



---

## 6. Security & Trust Architecture

1. **Secrets Management:** API keys in `.env`, excluded via `.gitignore`, referenced via `python-dotenv`.


2. **No PII in Synthetic Data:** Uses synthetic names and identifiers (`test_user_01@example.com`).


3. **Prompt Injection Defense:** Bank `narration` and ledger `customer` fields are attacker-controlled shapes. Sanitized before interpolation; explicit test on `TXN0050` confirms safety.


4. **Least-Privilege Agent Scope:** The agent returns a structured JSON verdict only; it never writes to the database directly or executes money movement. All state mutations occur in deterministic Python code.


5. **Immutable Audit Trail:** Every transaction reconciliation records `timestamp`, `input_snapshot`, `agent_verdict`, `confidence_score`, and `langfuse_trace_id`.


6. **Rate Limiting & Cost Guardrails:** Cap LLM calls per batch (e.g., max 50), log token spend via Langfuse, and fail gracefully if API errors occur.


7. **Input Validation:** Reject malformed CSV rows explicitly with Pydantic rather than silently coercing invalid numbers.



---

## 7. Vibe-Coding Implementation Roadmap

**Phase 1 — Schemas & Deterministic Matcher (Hours 0–4)**

* Define Pydantic models (`GatewayRow`, `BankRow`, `LedgerRow`, `ReconciliationVerdict`, `ExceptionRecord`).


* Implement `matcher/deterministic.py` (exact 3-way match, gross-to-net fee calculation, prefix UTR matching).


* Add unit tests (`tests/test_deterministic.py`) verifying all 33 clean matches offline.



**Phase 2 — LangGraph Agent & Langfuse Tracing (Hours 4–8)**

* Build LangGraph workflow (`CandidateFinder` $\to$ `LLMVerifier` $\to$ `DecisionGate`).


* Integrate Claude 3.5 Sonnet tool-calling with strict Pydantic output schemas.


* Wrap agent calls with Langfuse `@observe()` to track latency, tokens, and cost.

**Phase 3 — FastAPI Endpoints & Benchmark Evaluation (Hours 8–11)**

* Expose `/reconcile`, `/eval`, and `/audit` endpoints.


* Implement `evaluator.py` to benchmark Precision, Recall, and F1-score against `ground_truth.csv`.


* Verify injection handling on `TXN0050`.



**Phase 4 — Dashboard, Docker & Submission Artifacts (Hours 11–14)**

* Build Streamlit dashboard showing match rate %, exception triage queue, and live Langfuse audit log.


* Package stack into `Dockerfile` and `docker-compose.yml`.


* Record 5-minute video walkthrough (problem $\to$ architecture $\to$ live run on 53 records $\to$ exception review).



---

## 8. Deliverables Checklist (Mapped to Razorpay Evaluation Criteria)

| Criterion | Submission Artifacts |
| --- | --- |
| **Problem Taste** | Clear README framing real reconciliation friction, explaining why verification beats unbounded generation in fintech.

 |
| **Build Quality** | Clean repository layout, typed Pydantic models, deterministic fallback test suite, `.env.example`, working `docker-compose up`.

 |
| **AI Judgment** | Hybrid deterministic-first + LangGraph verifier architecture; explicit low-confidence exception routing.

 |
| **Failure Recovery** | Documented analysis of 3 handled failures: prompt injection containment, partial refund detection, and duplicate bank credit handling.

 |

---

## 9. Stretch Goals (Optional)

* **Hinglish/Free-text Narration Parser:** Extract intent and payment references from messy unstructured bank narrative logs.


* **Auto-Generated Ops Explanations:** A secondary LLM node generating human-readable resolution memos for finance teams.



```

```