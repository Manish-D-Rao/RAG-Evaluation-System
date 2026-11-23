import json
import os
import time
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("OPENAI_API_KEY", os.getenv("NEXUS_API_KEY", ""))
os.environ.setdefault("OPENAI_BASE_URL", os.getenv("NEXUS_BASE_URL", ""))

from deepeval.metrics import FaithfulnessMetric, GEval, AnswerRelevancyMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase, SingleTurnParams

MODEL = "gpt-5-nano"

with open("results2.jsonl", "r", encoding="utf-8") as file:
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

    answer_relevancy.measure(test_cases)
    time.sleep(7)
    faithfulness.measure(test_cases)
    time.sleep(7)
    context_relevancy.measure(test_cases)
    time.sleep(7)
    correctness.measure(test_cases)
    time.sleep(7)

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


with open("evaluated_results_v2.jsonl", "w", encoding="utf-8") as file:
    for result in evaluated_results:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")


print("\n\n Done. Results saved to evaluated_results_v2.jsonl")