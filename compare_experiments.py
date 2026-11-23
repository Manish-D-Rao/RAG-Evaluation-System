import json

experiments = []
for name in ["experiment_v1.jsonl", "experiment_v2_k2.jsonl", "experiment_v3_k4.jsonl", "experiment_v4_k4.jsonl"]:
    with open(name, "r", encoding="utf-8") as file:
        experiments.append(json.load(file))

print("\n========== EXPERIMENT COMPARISON ==========\n")

header = f"{'Metric':<20}"
for exp in experiments:
    header += f" {exp['experiment']:<20}"
print(header)
print("-" * len(header))

for metric in experiments[0]["metrics"]:
    row = f"{metric.replace('_', ' ').title():<20}"
    for exp in experiments:
        row += f" {exp['metrics'][metric]:<20.2f}"
    print(row)