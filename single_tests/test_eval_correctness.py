import os
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("OPENAI_API_KEY", os.getenv("NEXUS_API_KEY", ""))
os.environ.setdefault("OPENAI_BASE_URL", os.getenv("NEXUS_BASE_URL", ""))

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

test_case = LLMTestCase(
    input="How is compassion defined in the document?",
    actual_output="Compassion is defined here as genuine sympathy for others' suffering together with the wish to relieve it.",
    expected_output = "Compassion is genuine sympathy for others' suffering together with the wish to relieve it."

)

metric = GEval(
    model="gpt-5-nano",
    name="Correctness",
    criteria="The actual output should be factually equivalent to the expected output. And should have the same meaning, even if the wording is different.",
    evaluation_params = [
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT
    ]
)

metric.measure(test_case)

print(f"Score: {metric.score}")
print(f"Reason: {metric.reason}")