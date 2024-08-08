def write_response_to_file(response):
    f = open("output.txt", "a", encoding="utf-8")
    f.write(response)
    f.close()