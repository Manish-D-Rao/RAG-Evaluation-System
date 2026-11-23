import json
from statistics import mean

with open("evaluated_results_v4.jsonl", "r", encoding = "utf-8") as file:
    results = [json.loads(line) for line in file]


metrics = [
    "answer_relevancy",
    "faithfulness",
    "context_relevancy",
    "correctness"
]

print("\n============== Evaluation Summary ==============\n")

threshold = 0.7

for metric in metrics:
    scores = [result[metric] for result in results]

    avg_score = mean(scores)

    passed_count = sum(score >= threshold for score in scores)

    failed_count = len(scores) - passed_count

    passed_percentage = passed_count / len(scores) * 100

    print(metric.replace("_", " ").title() + ":")
    print(f"  Average Score: {avg_score:.2f}")
    print(f"         Passed: {passed_count}")
    print(f"         Failed: {failed_count}")
    print(f"Pass Percentage: {passed_percentage:.2f}%")
    print()