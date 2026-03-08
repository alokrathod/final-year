import json
import re
from modules.llm import call_llm

def extract_requirements(user_input):

    prompt = f"""
Extract atomic, clearly separable software requirements.

Rules:
- Split compound requirements.
- Preserve original meaning.
- No hallucination.

Return ONLY a JSON list with no markdown, no backticks, no explanation:

[
  {{"id": "REQ-001", "text": "..."}},
  {{"id": "REQ-002", "text": "..."}}
]

User Input:
{user_input}
"""

    response = call_llm(prompt)

    try:
        # Strip markdown code fences if Mistral wraps in ```json ... ```
        cleaned = re.sub(r"```json|```", "", response).strip()
        return json.loads(cleaned)
    except Exception as e:
        print(f"Extractor returned invalid JSON: {e}")
        print(response)
        return []   # always return a list, never None