import requests

OLLAMA_URL = "http://localhost:11434/api/generate"


def call_llm(prompt, model="mistral", temperature=0.2):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature
        }
    )

    if response.status_code != 200:
        raise Exception(f"LLM Error: {response.text}")

    return response.json()["response"].strip()