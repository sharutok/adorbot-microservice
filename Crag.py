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
from langchain_community.vectorstores import Chroma
from get_embedding_function import get_embedding_function
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv
from langsmith import Client
from langchain import hub
import json
from langchain_core.messages import HumanMessage, SystemMessage

openai_model = "gpt-4o-mini"

load_dotenv()


def query_rag(request_query, chroma_db, data_source):
    t1 = time.time()
    question = request_query

    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

    # chroma_db = "CHROMA_PDF"

    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=chroma_db, embedding_function=embedding_function)

    ### Retrieval Grader

    # Data model
    class GradeDocuments(BaseModel):
        """Binary score for relevance check on retrieved documents."""

        binary_score: str = Field(
            description="Documents are relevant to the question, 'yes' or 'no'"
        )

    # LLM with function call
    llm = ChatOpenAI(model=openai_model, temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Prompt
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
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

    docs = db.similarity_search_with_score(question, k=8)
    doc_txt = "\n\n---\n\n".join([doc.page_content for doc, _score in docs])
    sources = [doc.metadata.get("id", None) for doc, _score in docs]
    retrieval_grader.invoke({"question": question, "document": doc_txt})

    ### Generate
    # Prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                """You are an assistant for question-answering tasks. Strictly provide an answer in the context of Ador Welding LTD or ADOR, even if the user does not explicitly mention it.
                Use the following pieces of retrieved context to answer the question.\n
                If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n
                Question: {question} 
                Context: {context} 
                Answer (in the context of Ador Welding LTD or ADOR):""",
            ),
        ]
    )

    # LLM
    llm = ChatOpenAI(model_name=openai_model, temperature=0)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # System prompt to set the context for the LLM
    system = """You are a question re-writer that converts an input question to a better version that is optimized 
    for web search. Look at the input and try to reason about the underlying semantic intent / meaning.
    Stricty rewrite the question so that it is relevant to Ador Welding LTD or ADOR, even if the user does not explicitly mention it. 
    """

    # Create the prompt template for rewriting the question
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question that is strictly relevant to Ador Welding LTD or ADOR.",
            ),
        ]
    )

    # Initialize the output parser
    question_rewriter = re_write_prompt | llm | StrOutputParser()

    # Invoke the LLM with the input question
    rewritten_question = question_rewriter.invoke({"question": question})

    # Output the rewritten question
    question = rewritten_question

    ### Search

    # web_search_tool = TavilySearchResults(k=3)
    web_search_tool = TavilySearchResults(k=3, max_results=2)

    class GraphState(TypedDict):
        """
        Represents the state of our graph.

        Attributes:
            question: question
            generation: LLM generation
            web_search: whether to add search
            documents: list of documents
        """

        question: str
        generation: str
        web_search: str
        documents: List[str]

    def retrieve(state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        docs = db.similarity_search_with_score(question, k=8)
        return {"documents": docs, "question": question}

    def generate(state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        generation = rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    def grade_documents(state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        web_search = "No"
        score = retrieval_grader.invoke({"question": question, "document": documents})
        grade = score.binary_score

        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT 👍👍👍---")
            filtered_docs.append(documents)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT 😱😱😱---")
            web_search = "Yes"
        return {
            "documents": filtered_docs,
            "question": question,
            "web_search": web_search,
        }

    def transform_query(state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]

        # Re-write question
        better_question = question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question}

    def web_search(state):
        """
        Web search based on the re-phrased question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results
        """

        print("---WEB SEARCH---")
        question = state["question"]
        documents = state["documents"]

        # Web search
        docs = web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        documents.append(web_results)

        return {"documents": documents, "question": question}

    ### Edges

    _data_source = []

    def decide_to_generate(state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESSING GRADED DOCUMENTS---")
        state["question"]
        web_search = state["web_search"]
        state["documents"]

        if web_search == "Yes":
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            _data_source.append("web")
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            )
            return "transform_query"
        else:
            _data_source.append("database")
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE ---")
            return "generate"

    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generatae
    workflow.add_node("transform_query", transform_query)  # transform_query
    workflow.add_node("web_search_node", web_search)  # web search

    # Build graph
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "web_search_node")
    workflow.add_edge("web_search_node", "generate")
    workflow.add_edge("generate", END)

    # Compile
    app = workflow.compile()

    # Run
    inputs = {"question": question}
    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"Node '{key}':")
        pprint("\n---\n")

    # check if the response is from internet or local
    if _data_source[0]=="web":    
        system = """You are a response re-writer that gives a response by saying sorry i couldnt find data but this is what is found from internet.\n
        Strictly check if the question asked is relevent to Welding, Ador Welding LTD or ADOR, Welding Consumables, Electrodes, Wires, Fluxes, Product information,
        Product Specification or anything regarding welding if it is not relevent reply by saying to ask question regarding the relevent welding and others
        Always be concise and polite"""

        # Create the prompt template for rewriting the question
        re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Here is the initial response: \n\n {response} \n response by saying sorry i couldnt find data but this is what is found from internet. Always be concise and polite .",
                ),
            ]
        )

        question_rewriter = re_write_prompt | llm | StrOutputParser()

        # Invoke the LLM with the input question
        rewritten_question = question_rewriter.invoke({"response": value["generation"]})
        value["generation"]=rewritten_question

    t2 = time.time()
    print(value["generation"],"-----------",request_query)
    return json.dumps(
        {
            "questions": request_query,
            "ai_generated_questions": question,
            "response": value["generation"],
            "mess": "OK",
            "source_of_data": _data_source[0],
            "status_code": 200,
            "time_taken": "{} secs".format((t2 - t1)),
            "meta_response": {
                "sources": json.dumps(sources),
                # "question_rewriter": json.dumps(question_rewriter),
            },
        }
    )
