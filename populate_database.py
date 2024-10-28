import boto3
import os
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain.vectorstores.chroma import Chroma
from urllib.parse import urlparse
from utils import CHROMA
# CHROMA_PATH = "chroma"
# DATA_PATH = "data"


def main(chroma_db, data_source):
    try:
        # download_all_files_from_bucket()
        # Create (or update) the data store.
        # for PDF
        documents = load_documents_pdf(data_source)
        chunks = split_documents(documents)
        add_to_chroma(chunks,chroma_db)
    except Exception as e:
        print("error in main",e)

            
def load_documents_pdf(data_source):
    document_loader = PyPDFDirectoryLoader(data_source)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=2000,
        chunk_overlap=200,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document], data_source):
    # Load the existing database.
    db = Chroma(
        persist_directory=data_source, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):
    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id
    return chunks


def clear_database(chroma_db):
    if os.path.exists(chroma_db):
        shutil.rmtree(chroma_db)


def download_all_files_from_bucket():
    # Parse the S3 URI
    s3_uri = "s3://awsbuckettest001/adorbot_documents/"
    local_dir = "DATA_SOURCE_PDF"
    parsed_url = urlparse(s3_uri)
    bucket_name = parsed_url.netloc
    prefix = parsed_url.path.lstrip("/")

    # Initialize the S3 client
    session = boto3.Session(
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('REGION'),
    )
    # Initialize the S3 client using the session
    s3_client = session.client("s3")
    
    # Ensure the local directory exists
    os.makedirs(local_dir, exist_ok=True)

    try:
        # List objects in the bucket
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

        for page in pages:
            for obj in page.get("Contents", []):
                object_key = obj["Key"]
                local_file_path = os.path.join(
                    local_dir, os.path.relpath(object_key, prefix)
                )

                # Ensure the local directory for the current file exists
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                # Download the file from S3
                s3_client.download_file(bucket_name, object_key, local_file_path)
                print(f"Successfully downloaded {object_key} to {local_file_path}")
    except Exception as e:
        print(f"Error downloading files: {e}")
