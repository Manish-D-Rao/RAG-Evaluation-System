# RAG Evaluation System

A Retrieval-Augmented Generation (RAG) system built on a PDF document, evaluated across four metrics using DeepEval. This project demonstrates building a RAG pipeline, generating answers, and systematically measuring quality.

## Project Structure

```
├── rag1/                    # RAG pipeline
│   ├── file_reader.py       # PDF → paragraphs (PyMuPDF)
│   ├── indexing.py          # Paragraphs → embeddings → vector store
│   ├── tools.py             # Retrieval tool (similarity search, k=4)
│   ├── prompt.py            # System prompt for the agent
│   ├── objects.py           # Pydantic response schema (answer + context)
│   └── agent.py             # LLM agent (tool-calling, structured output)
├── eval_runner.py           # Runs all questions through RAG → results_v4.jsonl
├── eval_dataset.jsonl       # 25 Q&A pairs (ground truth)
├── eval_all_metrics_v3.py   # DeepEval evaluation (4 metrics per result)
├── save_experiments.py      # Aggregates scores into experiment summary
├── compare_experiments.py   # Side-by-side experiment comparison table
├── single_tests/            # Individual metric test scripts
└── experiment_v*.jsonl      # Experiment results (k=2, k=3, k=4)
```

## End-to-End Flow

```
PDF Document
    │
    ▼
┌─────────────────┐
│  file_reader.py │  PyMuPDF extracts text blocks → sorted by position → paragraphs
└────────┬────────┘
         │ 17 paragraphs (chunks)
         ▼
┌─────────────────┐
│  indexing.py    │  OpenAI text-embedding-3-small → InMemoryVectorStore
└────────┬────────┘
         │ vector store ready
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    eval_runner.py                           │
│  For each question in eval_dataset.jsonl:                  │
│    → ask_question()                                         │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                     rag1/agent.py                           │
│  LangChain agent with tool: collection_info                │
│  Model: gpt-6-luna (temperature=0)                         │
│                                                             │
│  Flow:                                                      │
│  1. Question → agent                                        │
│  2. Agent calls collection_info(query) tool                 │
│  3. Tool does similarity_search(k=4) on vector store       │
│  4. Retrieved chunks returned as context                    │
│  5. Agent generates answer grounded in context              │
│  6. Structured output: {answer, context}                    │
└────────────────────────────┬────────────────────────────────┘
                             │ results_v4.jsonl
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                 eval_all_metrics_v3.py                      │
│  For each result:                                           │
│    LLMTestCase(input, actual_output, expected_output,       │
│                retrieval_context)                           │
│    → AnswerRelevancyMetric                                  │
│    → FaithfulnessMetric                                     │
│    → ContextualRelevancyMetric                              │
│    → GEval (Correctness)                                    │
└────────────────────────────┬────────────────────────────────┘
                             │ evaluated_results_v4.jsonl
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              save_experiments.py → compare_experiments.py   │
│  Aggregate means → experiment_v*.jsonl → comparison table   │
└─────────────────────────────────────────────────────────────┘
```

## Component Deep Dive

### 1. PDF Reading (`file_reader.py`)
- Uses **PyMuPDF** (`fitz`) to extract text blocks from each page
- Blocks are sorted by position (top-to-bottom, left-to-right) to reconstruct reading order
- Line breaks (`\r`) within blocks are replaced with spaces
- Returns a flat list of paragraph strings — these become the chunks
- The PDF (Dalai Lama essay on the purpose of life) produces **17 chunks**

### 2. Indexing (`indexing.py`)
- Embeds all paragraphs using **OpenAI `text-embedding-3-small`** via the Nexus API proxy
- Stores in an **InMemoryVectorStore** (LangChain) — no external database needed
- This is a simple, fast setup suitable for a single-document RAG

### 3. Retrieval Tool (`tools.py`)
- Exposes `collection_info(query)` as a LangChain **tool** that the agent can call
- Performs **similarity search** with `k=4` (top 4 most similar chunks)
- Returns chunks joined by double newlines as context
- Also stores retrieved chunks in a global for the response's `context` field

### 4. Agent (`agent.py`)
- Built with **LangChain's `create_agent`** (tool-calling agent)
- Uses **gpt-6-luna** with `temperature=0` for deterministic, factual answers
- The agent decides when to call `collection_info` based on the system prompt
- Output is **structured** via Pydantic `AnswerResponse` (answer + context)
- `ask_question()` resets the global context, invokes the agent, returns `{answer, context}`

### 5. System Prompt (`prompt.py`)
- Instructs the agent to use `collection_info` before answering
- Grounds answers in retrieved context only
- Fallback: "pdf has no information about this question" if context is insufficient

### 6. Evaluation Dataset (`eval_dataset.jsonl`)
- **25 question-answer pairs** manually written from the PDF content
- Each entry: `{question, expected_answer}`
- Covers factual recall, reasoning, and inference questions

### 7. Evaluation Metrics (`eval_all_metrics_v3.py`)
Uses **DeepEval** with the Nexus LLM as the judge:

| Metric | What It Measures | Inputs Used |
|---|---|---|
| **Answer Relevancy** | Does the answer address the question? | input, actual_output |
| **Faithfulness** | Is the answer grounded in retrieved context? | actual_output, retrieval_context |
| **Contextual Relevancy** | Are retrieved chunks relevant to the question? | input, retrieval_context |
| **Correctness** (GEval) | Does the answer match the expected answer? | actual_output, expected_output |

Key implementation details:
- **Exponential backoff retry** on rate limit errors (30s → 60s → 120s → 240s)
- All metrics use `gpt-5-nano` as the judge model (separate from generator)
- Threshold: 0.7 for all metrics
- Results saved as JSONL with all scores per question

### 8. Experiment Tracking (`save_experiments.py`, `compare_experiments.py`)
- `save_experiments.py`: Reads evaluated results, computes **mean scores**, saves as `experiment_v*.jsonl`
- `compare_experiments.py`: Side-by-side comparison table of all experiments

## Experiments Run

| Experiment | Retriever k | Answer Rel | Faithfulness | Context Rel | Correctness |
|---|---|---|---|---|---|
| v1_k3 | 3 | 0.97 | 0.99 | 0.93 | 0.63 |
| v2_k2 | 2 | 0.99 | 0.99 | 0.91 | 0.56 |
| v3_k4 | 4 | 0.95 | 0.98 | 0.90 | 0.63 |
| v4_k4 | 4 | 0.97 | 0.99 | 0.98 | 0.64 |

**Key finding:** k=4 (v4) is best overall. k=2 drops correctness (too little context). k=3 and k=4 are similar, but v4 has better context relevancy.

## Key Concepts for Interviews

### Why RAG?
LLMs have knowledge cutoff and can't access private documents. RAG retrieves relevant context at query time and grounds the LLM's answer in it — reducing hallucinations and enabling domain-specific Q&A.

### Why an Agent instead of a simple RAG chain?
The agent uses **tool-calling** — it decides when to retrieve, can reformulate queries, and produces structured output. This is more flexible than a fixed retrieve-then-generate chain.

### Why 4 metrics?
No single metric captures RAG quality:
- **Context relevancy** measures the retriever
- **Faithfulness** measures grounding (no hallucination)
- **Answer relevancy** measures if the answer is on-topic
- **Correctness** measures factual accuracy vs ground truth

### Why temperature=0?
For factual Q&A, deterministic output is preferred — same question should always produce the same answer.

### Why separate generator and evaluator models?
Using the same model for generation and evaluation creates **self-grading bias** — the model favors its own style. Using gpt-6-luna for generation and gpt-5-nano for evaluation avoids this.

### What is noise dilution?
As k increases, more chunks are retrieved — but not all are relevant. Irrelevant chunks distract the LLM, reducing answer quality. This is why k=2 and k=4 can perform worse than k=3. Reranking or MMR retrieval can mitigate this.

### Rate limit handling
The Nexus API has token-per-minute limits. The eval script uses **exponential backoff** — when a 429 is hit, it waits 30s, then 60s, 120s, 240s before retrying. This keeps the eval running without manual intervention.

## How to Run

```bash
# 1. Generate RAG answers (produces results_v4.jsonl)
uv run python eval_runner.py

# 2. Evaluate with DeepEval (produces evaluated_results_v4.jsonl)
uv run python versions/v3/eval_all_metrics_v3.py

# 3. Save experiment summary (produces experiment_v4_k4.jsonl)
uv run python save_experiments.py

# 4. Compare all experiments
uv run python compare_experiments.py
```

## Tech Stack

- **LangChain** — agent, tools, vector store abstraction
- **PyMuPDF (fitz)** — PDF text extraction
- **OpenAI SDK** — embeddings + LLM calls (via Nexus proxy)
- **DeepEval** — RAG evaluation framework
- **Pydantic** — structured output schema
- **python-dotenv** — environment variable management
- **uv** — dependency management