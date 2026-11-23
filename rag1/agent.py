import os
import rag1.tools as tl
from .prompt import system_agent_prompt
from langchain_openai import ChatOpenAI
from .tools import collection_info
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from .objects import AnswerResponse
from dotenv import load_dotenv

load_dotenv()
NEXUS_API_KEY = os.getenv("NEXUS_API_KEY")
NEXUS_BASE_URL = os.getenv("NEXUS_BASE_URL")

llm = ChatOpenAI(model="gpt-6-luna", temperature=0, api_key=NEXUS_API_KEY, base_url=NEXUS_BASE_URL)

agent = create_agent(
    model=llm,
    tools=[collection_info],
    system_prompt=system_agent_prompt,
    response_format=AnswerResponse,
)

def ask_question(question: str) -> dict:
    """
    Ask a question to the agent and return the answer along with the context used to generate the answer.
    """

    tl.last_retrieved_context = []  # Reset the last retrieved context before each question

    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]}
    )

    structured_response = result["structured_response"]

    return {
        "answer": structured_response.answer,
        "context" : structured_response.context
    }

# print(ask_question("What is the summary of the pdf?"))