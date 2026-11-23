from rag1.agent import ask_question
import json

with open("eval_dataset.jsonl", "r", encoding="utf-8") as file:
    test_cases = [json.loads(line) for line in file]

    results = []

    for test_case in test_cases:
        question = test_case["question"]
        expected_answer = test_case["expected_answer"]

        print(f"Question: {question}")

        rag_response = ask_question(question)

        result = {
            "question": question,
            "expected_answer": expected_answer,
            "rag_answer": rag_response["answer"],
            "context": rag_response["context"]
        }

        results.append(result)

        print("RAG Answer:", rag_response["answer"])

with open("results_v4.jsonl", "w", encoding="utf-8") as file:
    for result in results:
            file.write(json.dumps(result) + "\n")


print("Evaluation completed. Results saved to results_v4.jsonl.")
