import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"

def call_llm(prompt, temperature=0.2):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature
        }
    )

    if response.status_code != 200:
        raise Exception(f"LLM Error: {response.text}")

    return response.json()["response"].strip()