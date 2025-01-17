import time
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from get_embedding_function import get_embedding_function
from langchain_openai import ChatOpenAI
from flask import request
from flask import Flask
from populate_database import main
from populate_database import clear_database
from utils import write_response_to_file
import os
from dotenv import load_dotenv
import requests
import json
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    jwt_required,
    get_jwt_identity,
)


CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
---
Answer the question based on the above context: {question}
"""


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=7)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"{response_text} {sources}"
    write_response_to_file("questions" + query_text + "\n")
    write_response_to_file("response" + formatted_response + "\n")
    write_response_to_file("---------END-----------" + "\n")
    return formatted_response


app = Flask(__name__)
app.config["JWT_SECRET_KEY"] = os.environ("JWT_SECRET_KEY")
jwt = JWTManager(app)

@app.route('/generate-jwt',)


@app.route("/generate/text/", methods=["POST"])
def hello_world():
    t1 = time.time()

    request_query = request.json["questions"]
    response = query_rag(request_query)
    data = ((response.split("response_metadata")[0]).split("=")[1]).replace("'", "")
    t2 = time.time()
    print("{} secs".format((t2 - t1)))

    save_to_db(
        questions=request_query,
        _response=data.replace("\\n", "\n"),
        response_status=False,
        meta_response=response
        )

    return {
        "question": request_query,
        "response": data.replace("\\n", "\n"),
        "mess": "OK",
        "status_code": 200,
        "meta_response":response
        }


@app.route("/add-files/", methods=["POST"])
def add_files():
    try:
        main()
        return {
            "status_code": 200,
        }
    except Exception as e:
        return {
            "error": e,
            "status_code": 400,
        }

@app.route("/clear/db",methods=["GET"])
def clean_database():
    try:
        print("✨ Clearing Database")
        clear_database()
        return {
            "status_code": 200,
        }
    except Exception as e:
        return {
            "error": e,
            "status_code": 400,
        }


def save_to_db(questions, _response, response_status, meta_response):
    try:
        url = "http://localhost:8000/conv/20283e81-65be-4106-818f-f015bb67a10f/"

        payload = json.dumps(
            {
                "user_id": "20283e81-65be-4106-818f-f015bb67a10f",
                "questions": questions,
                "response": _response,
                "response_status": response_status,
                "other_info": meta_response,
            }
        )
        headers = {"Content-Type": "application/json"}
        response = requests.request("POST", url, headers=headers, data=payload)
        print(response.text)        
    except Exception as e:
        print("error in save_to_db",e)
