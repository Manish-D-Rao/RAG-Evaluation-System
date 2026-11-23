# RAG Evaluation Metrics

This document describes the four metrics used to evaluate the RAG (Retrieval-Augmented Generation) system in this project.

## 1. Answer Relevancy

Measures how well the generated answer addresses the user's question. An answer is relevant if it directly responds to the query without drifting into unrelated topics. Even a factually correct answer can score poorly if it fails to answer what was asked.

- **Focus:** Does the answer match the intent of the question?
- **Failure mode:** Off-topic, evasive, or overly generic responses.

## 2. Correctness

Measures whether the generated answer matches the expected/ground-truth answer. It checks factual accuracy and completeness against a reference answer (see `eval_dataset.jsonl`'s `expected_answer` field).

- **Focus:** Is the answer factually right and complete?
- **Failure mode:** Hallucinations, missing key facts, or contradictions with the reference.

## 3. Faithfulness

Measures whether the generated answer is grounded in the retrieved context. Every claim in the answer should be supported by the retrieved chunks; nothing should be invented beyond what the context provides.

- **Focus:** Is the answer faithful to the retrieved documents?
- **Failure mode:** Hallucinated details not present in the retrieved context.

## 4. Context Relevancy

Measures whether the retrieved context chunks are relevant to the user's question. It evaluates the quality of the retrieval step, ensuring that the chunks fed to the generator actually contain information needed to answer the query.

- **Focus:** Are the retrieved chunks useful for answering the question?
- **Failure mode:** Irrelevant or noisy chunks diluting the context window.

## Summary

| Metric | Stage | Question it answers |
|---|---|---|
| Context Relevancy | Retrieval | Did we fetch the right chunks? |
| Faithfulness | Generation | Did the answer stay within the context? |
| Answer Relevancy | Generation | Did the answer address the question? |
| Correctness | End-to-end | Does the answer match the ground truth? |
