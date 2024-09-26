import os
# from pprint import pprint
# import time
# from langgraph.graph import END, StateGraph, START
# from langchain.schema import Document
# from typing import List
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain import hub
# from langchain_core.output_parsers import StrOutputParser
# from typing_extensions import TypedDict
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.pydantic_v1 import BaseModel, Field
# from langchain_openai import ChatOpenAI
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from get_embedding_function import get_embedding_function
# from langchain.
# vectorstores.chroma import Chroma
from dotenv import load_dotenv
# from langsmith import Client
# from langchain import hub
# import json

load_dotenv()


# def query_rag():
#     t1 = time.time()
#     question = "what is diffrent type of welding positions"

    # os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    # os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

#     class GradeDocuments(BaseModel):
#         """Binary score for relevance check on retrieved documents."""

#         binary_score: str = Field(
#             description="Documents are relevant to the question, 'yes' or 'no'"
#         )

#     # LLM with function call
#     llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
#     structured_llm_grader = llm.with_structured_output(GradeDocuments)

#     # Prompt
#     system = """You are a grader assessing relevance of a data.\n
#         If the data contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
#         Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
#     grade_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", system),
#             (
#                 "human",
#                 "Retrieved document: \n\n {document} \n\n User question: {question}",
#             ),
#         ]
#     )

#     retrieval_grader = grade_prompt | structured_llm_grader
#     question = question
#     chat_history = [
#         [
#             "what is llm",
#             "LLM stands for Large Language Model. Its applicati…s to enhance interactions and automate processes.",
#         ],
#         [
#             "what is GWAM",
#             "GWAM stands for Gas Metal Arc Welding, which is an arc welding process where an electric arc is formed between a continuous consumable electrode and the workpiece. It utilizes an external supply of shielding gas to protect the molten weld pool, allowing for higher deposition rates and minimal post-weld cleaning. This process is also known by its subtypes, such as Metal Inert Gas (MIG) welding and Metal Active Gas (MAG) welding, depending on the shielding gas used.",
#         ],
#         [
#             "what is MAM",
#             "MAM stands for Mobile Application Management. Its applications include securing, managing, and distributing mobile applications in enterprise settings, as well as controlling and monitoring business-related data on both company-owned and BYOD devices. MAM allows businesses to enforce application management policies and remove unauthorized access to sensitive data.",
#         ],
#         [
#             "what is diffrent type of welding positions",
#             "Different types of welding positions include flat, horizontal, vertical, and overhead positions. Each position affects the welding technique and the way the welder must manipulate the welding equipment. Specific procedures and certifications may be required for welders to perform in these various positions.",
#         ],
#     ]
#     _chat_history=[]
#     for chat in chat_history:
#         _response = retrieval_grader.invoke({"question": question, "document": chat[1]})
#         print(_response.binary_score)
#         if "yes" in _response.binary_score:
#             _chat_history.append(chat)

#     print(_chat_history)

# query_rag()


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Question to re-write
question = "what llm?"

# Initialize the ChatOpenAI model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# System prompt to set the context for the LLM
system = """You are a question re-writer that converts an input question to a better version that is optimized 
for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""

# Create the prompt template for rewriting the question
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

# Initialize the output parser
question_rewriter = re_write_prompt | llm | StrOutputParser()

# Invoke the LLM with the input question
rewritten_question = question_rewriter.invoke({"question": question})

# Output the rewritten question
print(rewritten_question)
