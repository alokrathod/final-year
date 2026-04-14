from modules.llm import call_llm
import json


def analyze_system(srs_text):

    prompt = f"""
You are a software architect.

Analyze the following SRS and extract key system characteristics.

Identify:

1. System type (web app, mobile app, API, etc.)
2. Core features (list 5–10 major functionalities)
3. Non-functional requirements (performance, security, scalability, etc.)
4. Estimated complexity (low, medium, high)

Return ONLY JSON:

{{
  "system_type": "...",
  "features": ["...", "..."],
  "non_functional": ["...", "..."],
  "complexity": "..."
}}

SRS:
{srs_text}
"""

    response = call_llm(prompt, model="mistral", temperature=0.2)

    try:
        data = json.loads(response)
    except:
        print("⚠️ Failed to parse system analysis JSON")
        print(response)
        return {}

    return data