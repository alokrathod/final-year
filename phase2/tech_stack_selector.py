from modules.llm import call_llm
import json


def select_tech_stack(system_analysis, architecture):

    prompt = f"""
You are a senior software architect.

Based on the system analysis and architecture below, select a suitable tech stack.

System Analysis:
{system_analysis}

Architecture:
{architecture}

Tasks:
1. Choose frontend technology
2. Choose backend technology
3. Choose database
4. Optionally suggest tools/frameworks
5. Justify your choices

Rules:
- If not specified, infer using best practices
- Ensure technologies are compatible

Return ONLY JSON:

{{
  "frontend": "...",
  "backend": "...",
  "database": "...",
  "tools": ["...", "..."],
  "justification": "..."
}}
"""

    response = call_llm(prompt, model="mistral", temperature=0.2)

    # clean response
    response = response.strip().replace("```json", "").replace("```", "")

    try:
        data = json.loads(response)
    except:
        print("⚠️ Failed to parse tech stack JSON")
        print(response)
        return {}

    return data