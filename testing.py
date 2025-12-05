import requests
import json

ai_url = "http://127.0.0.1:8000/perguntar"

question = {
    "question_text": "Qual o Salmo mais lindo?"
}

try:
    response = requests.post(url=ai_url, json=question)

    if response.status_code == 200:
        answer_data = response.json()
        print(answer_data['resposta'])
    else:
        print(f"Erro ao chamar IA. {response.status_code}")
        print(response.text)

except requests.exceptions.ConnectionError:
    print("Não conectou")
    print("lembra de rodar 'uvicorn RAG:app --reload'")