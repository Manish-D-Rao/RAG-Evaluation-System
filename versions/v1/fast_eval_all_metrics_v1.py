import json
import os
import time
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("OPENAI_API_KEY", os.getenv("NEXUS_API_KEY", ""))
os.environ.setdefault("OPENAI_BASE_URL", os.getenv("NEXUS_BASE_URL", ""))

from deepeval.metrics import FaithfulnessMetric, GEval, AnswerRelevancyMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.evaluate import CacheConfig, AsyncConfig
from deepeval import evaluate

MODEL = "gpt-5-nano"

with open("results1.jsonl", "r", encoding="utf-8") as file:
    results = [json.loads(line) for line in file if line.strip()]

question_test = []

for i, result in enumerate(results, start=1):
    print(f"\nEvaluating {i}/{len(results)}")

    context = result["context"]

    if isinstance(context, str):
        retrieval_context = [context]
    else:
        retrieval_context = context

    test_case = LLMTestCase(
            input=result["question"],
            actual_output=result["rag_answer"],
            expected_output=result["expected_answer"],
            retrieval_context=retrieval_context,
        )

    question_test.append(test_case)

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
    threshold = 0.7
)

evaluation_results = evaluate(
    test_cases = question_test,
    metrics = {
        answer_relevancy,
        faithfulness,
        context_relevancy,
        correctness
    },
    cache_config = CacheConfig(
        use_cache = False,
        write_cache = False
    ),
    async_config = AsyncConfig(
        run_async = True,
        max_concurrent = 1,
        throttle_value = 12
    )
)

evaluated_results = []
for i, (original_result, test_result) in enumerate(zip(results, evaluation_results.test_results), start = 1):
    
    answer_relevancy_score = None
    faithfulness_score = None
    context_relevancy_score = None
    correctness_score = None

    print("\nMetrics:")

    for metric in test_result.metrics_data or []:
        metric_name = metric.name.strip().lower()

        if metric_name == "answer_relevancy":
            answer_relevancy_score = metric.score
        elif metric_name == "faithfulness":
            faithfulness_score = metric.score
        elif metric_name == "context_relevancy":
            context_relevancy_score = metric.score
        elif metric_name.startswith("correctness"):
            correctness_score = metric.score

    evaluated_result = {
            **result,
            "answer_relevancy": answer_relevancy_score,
            "faithfulness": faithfulness_score,
            "context_relevancy": context_relevancy_score,
            "correctness": correctness_score
        }

    evaluated_results.append(evaluated_result)

with open("evaluated_results1_FAST.jsonl", "w", encoding="utf-8") as file:
    for result in evaluated_results:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")


print("\n\n Done. Results saved to evaluated_results1.jsonl")