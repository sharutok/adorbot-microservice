import json
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request
from get_embedding_function import get_embedding_function
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from populate_database import clear_database, main
from query_data import query_rag
# from Crag import query_rag
from utils import CHROMA, DATA_SOURCE, write_response_to_file
app = Flask(__name__)

@app.route("/generate/text/", methods=["POST"])
def hello_world():
    t1 = time.time()
    request_query = request.json["questions"]
    chat_history=request.json.get("chat_history", [])
    which_db = "PDF"
    if which_db in list(CHROMA):
        chroma_db = CHROMA[which_db]
        data_source = DATA_SOURCE[which_db]
        # response = query_rag(request_query, chroma_db, data_source,chat_history)
        response = query_rag(request_query, chroma_db, data_source,chat_history)
        t2 = time.time()
        print("{} secs".format((t2 - t1)))
    return response

@app.route("/add-files/", methods=["POST"])
def add_files():
    which_db = request.json["which_db"]
    if which_db in list(CHROMA):
        try:
            chroma_db = CHROMA[which_db]
            data_source = DATA_SOURCE[which_db]
            main(chroma_db, data_source)
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


@app.route("/clear/db", methods=["GET"])
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

@app.route("/health/check",methods=["GET"])
def healthCheck():
    try:
        return {"status_code": 200}
    except Exception as e:
          return {"status_code": 400}