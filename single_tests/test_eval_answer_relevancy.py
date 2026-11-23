import os
from dotenv import load_dotenv

load_dotenv()

# DeepEval reads OpenAI config from these env vars; point them at the Nexus endpoint.
os.environ.setdefault("OPENAI_API_KEY", os.getenv("NEXUS_API_KEY", ""))
os.environ.setdefault("OPENAI_BASE_URL", os.getenv("NEXUS_BASE_URL", ""))

from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="What bring the most happiness to humans?",
    actual_output=("Love and compassion bring the most happiness to humans, because human nature values them above all else and they bring inner peace."),
)

metric = AnswerRelevancyMetric(threshold=0.7, model="gpt-5-nano")

metric.measure(test_case)

print(f"Score: {metric.score}")
print(f"Reason: {metric.reason}")
print(f"Passed: {metric.is_successful()}")


