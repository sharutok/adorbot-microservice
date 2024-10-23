import requests


import pandas as pd
import json

# Path to the Excel file
file_path = "questions.xlsx"

# Read the Excel file
df = pd.read_excel(file_path)

json_data = df.to_json(orient="records")


val=json.loads( json_data)

questions=[]
for a in val:
    questions.append(a["Q"])

print(len(questions))

url = "http://localhost:5000/generate/text"

if True:
    for q in questions:
        data = {"questions": q, "which_db": "PDF"}

        response = requests.post(url, json=data)

        if response.status_code == 200:
            generated_text = response.json()
            print("Generated Text:", generated_text)
        else:
            print("Failed to call the API:", response.status_code)