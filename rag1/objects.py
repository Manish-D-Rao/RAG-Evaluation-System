from pydantic import BaseModel, Field

class AnswerResponse(BaseModel):
    answer: str = Field(description="The answer to the question based on the PDF content.")
    context: str = Field(description="The context  retrieved from the PDF that agent used to generate the answer.")

