from datetime import datetime
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
import json
from utils import CHROMA,DATA_SOURCE
from langchain.schema.output_parser import StrOutputParser


PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks.Answer the question based only on the following context:
{context}
---
Answer the question based on the above context: {question}
"""

def query_rag(query_text: str,chroma_db,data_source):
    # Prepare the DB.
    
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=chroma_db, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=7)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    response_text = model.invoke(prompt)
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"{response_text} {sources}"

    write_response_to_file("Time " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
    write_response_to_file("questions " + query_text + "\n")
    write_response_to_file("response " + formatted_response + "\n")
    write_response_to_file("---------END-----------" + "\n")

    data = ((formatted_response.split("response_metadata")[0]).split("=")[1]).replace("'", "")
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


app = Flask(__name__)

@app.route("/generate/text/", methods=["POST"])
def hello_world():
    t1 = time.time()

    request_query = request.json["questions"]
    which_db = request.json["which_db"]
    if which_db in list(CHROMA):
        chroma_db=CHROMA[which_db]
        data_source = DATA_SOURCE[which_db]
        response = query_rag(request_query,chroma_db,data_source)
        
        data = ((response.split("response_metadata")[0]).split("=")[1]).replace("'", "")
        t2 = time.time()
        print("{} secs".format((t2 - t1)))

        return json.dumps({
            "questions": request_query,
            "response": data.replace("\\n", "\n"),
            "mess": "OK",
            "status_code": 200,
            "meta_response":response
            })

@app.route("/add-files/", methods=["POST"])
def add_files():
    which_db = request.json["which_db"]
    if which_db in list(CHROMA):
        try:
            chroma_db=CHROMA[which_db]
            data_source = DATA_SOURCE[which_db]
            main(chroma_db,data_source)
            return {
                "status_code": 200,
            }
        except Exception as e:
            print(e)
            return {
                "status_code": 400,
            }
    return {
        "status_code": 401,
    }

@app.route("/clear/db",methods=["GET"])
def clean_database():
    which_db = request.json["which_db"]
    if which_db in list(CHROMA):
        try:
            print("✨ Clearing Database {}".format(which_db))
            chroma_db = CHROMA[which_db]
            clear_database(chroma_db)
            return {
                "status_code": 200,
            }
        except Exception as e:
            print(e)
            return {
                # "error": e,
                "status_code": 400,
            }
    return {
        "status_code": 401,
    }
