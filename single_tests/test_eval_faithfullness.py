import os
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("OPENAI_API_KEY", os.getenv("NEXUS_API_KEY", ""))
os.environ.setdefault("OPENAI_BASE_URL", os.getenv("NEXUS_BASE_URL", ""))

from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="How is compassion defined in the document?",
    actual_output="Compassion is defined here as genuine sympathy for others' suffering together with the wish to relieve it.",
    retrieval_context=[
        "The more we care about the happiness of others, the greater our own well-being. A warm-hearted attitude toward others calms the mind, reduces fear and insecurity, and gives strength to face obstacles. When problems arise, remembering that everyone suffers gives a more realistic perspective, and each difficulty becomes a chance to train the mind. Compassion is defined here as genuine sympathy for others' suffering together with the wish to relieve it.",
    ],
)

metric = FaithfulnessMetric(threshold=0.7, model="gpt-5-nano")

metric.measure(test_case)

print(f"Score: {metric.score}")
print(f"Reason: {metric.reason}")
print(f"Passed: {metric.is_successful()}")