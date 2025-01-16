from datetime import datetime
import time
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from get_embedding_function import get_embedding_function
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from flask import request
from flask import Flask
from populate_database import main
from populate_database import clear_database
from utils import write_response_to_file
import os
from dotenv import load_dotenv
import json
from utils import CHROMA, DATA_SOURCE
from langchain.schema.output_parser import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from openai import OpenAI
import requests

load_dotenv()
from grade_chat_history import grade_chat_history

# PROMPT_TEMPLATE = """
# You are an assistant for question-answering tasks.Answer the question based only on the following context:
# {context}
# ---
# Answer the question based on the above context: {question}
# """

PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question considering the history of the conversation. 
If the context provides information to answer the question, respond with a concise answer in three sentences maximum using that information.
If the context does not provide information, respond strictly by replying NO .
Chat history: {chat_history}
Question: {question} 
Context: {context} 
"""
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


def query_rag(query_text: str, chroma_db, data_source, chat_history):
    if check_question_asked_type(query_text):
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=chroma_db, embedding_function=embedding_function)

        # Search the DB.
        results = db.similarity_search_with_score(query_text, k=7)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

        corrected_chat_history = (
            grade_chat_history(query_text, chat_history) if len(chat_history) else []
        )

        # corrected chat history
        _corrected_chat_history = ""
        _corrected_chat_history = (
            corrected_chat_history if corrected_chat_history else "No history"
        )
        # print(_corrected_chat_history)
        prompt = prompt_template.format(
            context=context_text, question=query_text, chat_history=_corrected_chat_history
        )

        model = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
        response_text = model.invoke(prompt)
        sources = [doc.metadata.get("id", None) for doc, _score in results]
        formatted_response = f"{response_text} {sources}"

        write_response_to_file(
            "Time " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n"
        )
        write_response_to_file("questions " + query_text + "\n")
        write_response_to_file("response " + formatted_response + "\n")
        write_response_to_file("---------END-----------" + "\n")

        data = ((formatted_response.split("response_metadata")[0]).split("=")[1]).replace(
            "'", ""
        )
        ####### SEARCH FROM WEB #######
        print(data.replace("\\n", "\n"))
        if "NO" in data.replace("\\n", "\n"):
            print("--------------SEARCHING FOR WEB--------------------")
            searched_internet=True
            web_search_tool = TavilySearchResults(k=3, max_results=2)
            docs = web_search_tool.invoke({"query": query_text})
            web_results = "\n".join([d["content"] for d in docs])

            prompt = prompt_template.format(
                context=web_results,
                question=query_text,
                chat_history=_corrected_chat_history,
            )
            response_text = model.invoke(prompt)
            formatted_response = f"{response_text} {sources}"
            data = (
                (formatted_response.split("response_metadata")[0]).split("=")[1]
            ).replace("'", "")
        else:
            searched_internet=False
            print("--------------SEARCHED LOCALLY--------------------")

        # return json.dumps(
        #     {
        #         "questions": query_text,
        #         "response": data.replace("\\n", "\n"),
        #         "mess": "OK",
        #         "status_code": 200,
        #         "meta_response": formatted_response,
        #     }
        # )
        return json.dumps(
            {
                "searched_internet":searched_internet,
                "questions": query_text,
                "ai_generated_questions": query_text,
                "response": data.replace("\\n", "\n"),
                "mess": "OK",
                "source_of_data": formatted_response,
                "status_code": 200,
                "meta_response": {
                    "sources": "",
                },
            }
        )
    else:
        print("searching for pdf!!!")
        data=call_qa_microservice(query_text)
        return json.dumps(
            {
                "searched_internet":False,
                "questions": query_text,
                "ai_generated_questions": query_text,
                "response": (json.loads(data))['data'],
                "mess": "OK",
                "source_of_data": "",
                "status_code": 200,
                "meta_response": {
                    "sources": "",
                },
            }
        )
        


app = Flask(__name__)


@app.route("/generate/text/", methods=["POST"])
def hello_world():
    try:
        t1 = time.time()

        request_query = request.json["questions"]

        if check_question_asked_type(request_query):
            which_db = request.json["which_db"]
            if which_db in list(CHROMA):
                chroma_db = CHROMA[which_db]
                data_source = DATA_SOURCE[which_db]
                response = query_rag(request_query, chroma_db, data_source)
                data = ((response.split("response_metadata")[0]).split("=")[1]).replace("'", "")

            t2 = time.time()
            print("{} secs".format((t2 - t1)))
            return json.dumps(
                {
                    "questions": request_query,
                    "response": data.replace("\\n", "\n"),
                    "mess": "OK",
                    "status_code": 200,
                    "meta_response": response,
                }
            )
        else:
            data=call_qa_microservice(request_query)
            t2 = time.time()
            print("{} secs".format((t2 - t1)))
            return json.dumps(
                {
                    "questions": request_query,
                    "response": data,
                    "mess": "OK",
                    "status_code": 200,
                    "meta_response": response,
                }
            )


    except Exception as e:
        print("Error in hello_world, route->/generate/text/",e)


# @app.route("/add-files/", methods=["POST"])
# def add_files():
#     which_db = request.json["which_db"]
#     if which_db in list(CHROMA):
#         try:
#             chroma_db = CHROMA[which_db]
#             data_source = DATA_SOURCE[which_db]
#             main(chroma_db, data_source)
#             return {
#                 "status_code": 200,
#             }
#         except Exception as e:
#             print(e)
#             return {
#                 "status_code": 400,
#             }
#     return {
#         "status_code": 401,
#     }


# @app.route("/clear/db", methods=["GET"])
# def clean_database():
#     which_db = request.json["which_db"]
#     if which_db in list(CHROMA):
#         try:
#             print("✨ Clearing Database {}".format(which_db))
#             chroma_db = CHROMA[which_db]
#             clear_database(chroma_db)
#             return {
#                 "status_code": 200,
#             }
#         except Exception as e:
#             print(e)
#             return {
#                 # "error": e,
#                 "status_code": 400,
#             }
#     return {
#         "status_code": 401,
#     }


def check_question_asked_type(questions):
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You will be provided with a block of text, and your task is to search for keywords like pdf, link and any other synonyms and reply only True if present and False if not",
            },
            {"role": "user", "content": questions},
        ],
        temperature=0.5,
        max_tokens=256,
        top_p=1,
        response_format={"type": "text"},
    )
    if (response.choices[0].message.content=="True"):
        print("is a pdf")
        return False
    else:
        print("is not a pdf")
        return True

# check_question_asked_type("provide me weblink of AUSTOMANG-307")


def call_qa_microservice(questions):
    url = "{}/query".format(os.getenv("QA_MICROSERVICE_BACKEND_URL"))
    payload = json.dumps({"query": questions})
    headers = {'Content-Type': 'application/json'}
    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)
    return response.text
