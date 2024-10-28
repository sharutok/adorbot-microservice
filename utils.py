CHROMA = {
    "PDF": "CHROMA_PDF",
    "USER_TRAIN_DATA": "CHROMA_USER_TRAIN_DATA",
    "USER_RESPONSE_DATA": "CHROMA_USER_RESPONSE_DATA",
}

DATA_SOURCE = {
    "PDF": "DATA_SOURCE_PDF",
    "USER_TRAIN_DATA": "DATA_SOURCE_USER_TRAIN_DATA",
    "USER_RESPONSE_DATA": "DATA_SOURCE_USER_RESPONSE_DATA",
}

def write_response_to_file(response):
    f = open("output.txt", "a", encoding="utf-8")
    f.write(response)
    f.close()


