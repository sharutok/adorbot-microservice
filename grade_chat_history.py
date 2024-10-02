import os
from pprint import pprint
import time
from langgraph.graph import END, StateGraph, START
from langchain.schema import Document
from typing import List
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from get_embedding_function import get_embedding_function
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv
from langsmith import Client
from langchain import hub
import json
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains import LLMChain

load_dotenv()



def grade_chat_history(_chat_history_):

    #RELIVANCE GRADER OF QUERY WITH CHAT HISTORY

    question = "what is diffrent type of welding positions"

    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

    class GradeDocuments(BaseModel):
        """Binary score for relevance check on retrieved documents."""

        binary_score: str = Field(
            description="Documents are relevant to the question, 'yes' or 'no'"
        )

    # LLM with function call
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Prompt
    system = """You are a grader assessing relevance of a data.\n
        If the data contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Retrieved document: \n\n {document} \n\n User question: {question}",
            ),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader
    question = question
    chat_history = _chat_history_
    _chat_history=[]
    for chat in chat_history:
        _response = retrieval_grader.invoke({"question": question, "document": chat[1]})
        print(_response.binary_score)
        if "yes" in _response.binary_score:
            _chat_history.append(chat)

    print(_chat_history)




prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant. Given a chat history and the latest user question, 
            use the context from the chat history if it's relevant to the question. 
            Otherwise, answer based on the question provided.""",
        ),
        MessagesPlaceholder(
            variable_name="chat_history"
        ),  
        ("human", "{question}"), 
    ]
)

# Sample question
question = "what are all the industries are being used?"

# Example chat history with prior messages
chat_history = [
    HumanMessage(content="what is GMAM welding?"),
    SystemMessage(
        content="Gas Metal Arc Welding (GMAW) is an arc welding process where an electric arc is formed between a continuous consumable electrode and the workpiece, with shielding provided by an external gas or gas mixture. The process allows for higher deposition rates compared to shielded metal arc welding and requires minimal post-weld cleaning due to the absence of fused slag. GMAW can operate using different shielding gases, leading to its subtypes known as MIG (Metal Inert Gas) and MAG (Metal Active Gas) welding."
    ),
]

# LLM setup
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define the LLM Chain
llm_chain = LLMChain(llm=llm, prompt=prompt, output_parser=StrOutputParser())

# Invoke the chain
generation = llm_chain.invoke({"chat_history": chat_history, "question": question})

# Output the generated response
print(generation["text"])
