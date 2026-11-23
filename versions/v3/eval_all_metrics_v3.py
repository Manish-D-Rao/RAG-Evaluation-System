import json
import os
import time
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("OPENAI_API_KEY", os.getenv("NEXUS_API_KEY", ""))
os.environ.setdefault("OPENAI_BASE_URL", os.getenv("NEXUS_BASE_URL", ""))

from openai import RateLimitError
from deepeval.metrics import FaithfulnessMetric, GEval, AnswerRelevancyMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase, SingleTurnParams

MODEL = "gpt-5-nano"


def measure_with_retry(metric, test_case, max_retries=5, base_delay=30):
    """Measure a metric, retrying with exponential backoff on rate limit."""
    for attempt in range(max_retries):
        try:
            metric.measure(test_case)
            return
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"  Rate limited — waiting {delay}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)


with open("results_v3.jsonl", "r", encoding="utf-8") as file:
    results = [json.loads(line) for line in file if line.strip()]

evaluated_results = []

for i, result in enumerate(results, start=1):
    print(f"\nEvaluating {i}/{len(results)}")

    test_cases = LLMTestCase(
        input=result["question"],
        actual_output=result["rag_answer"],
        expected_output=result["expected_answer"],
        retrieval_context=[result["context"]],
    )

    answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model=MODEL)
    faithfulness = FaithfulnessMetric(threshold=0.7, model=MODEL)
    context_relevancy = ContextualRelevancyMetric(threshold=0.7, model=MODEL)
    correctness = GEval(
        model=MODEL,
        name="Correctness",
        criteria="The actual output should be factually equivalent to the expected output. And should have the same meaning, even if the wording is different.",
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
    )

    measure_with_retry(answer_relevancy, test_cases)
    measure_with_retry(faithfulness, test_cases)
    measure_with_retry(context_relevancy, test_cases)
    measure_with_retry(correctness, test_cases)

    evaluated_result = {
        **result,
        "answer_relevancy": answer_relevancy.score, "faithfulness": faithfulness.score,
        "context_relevancy": context_relevancy.score, "correctness": correctness.score
    }

    evaluated_results.append(evaluated_result)

    print("answer_relevancy", answer_relevancy.score)
    print("faithfulness", faithfulness.score)
    print("context_relevancy", context_relevancy.score)
    print("correctness", correctness.score)


with open("evaluated_results_v3.jsonl", "w", encoding="utf-8") as file:
    for result in evaluated_results:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")


print("\n\n Done. Results saved to evaluated_results_v3.jsonl")