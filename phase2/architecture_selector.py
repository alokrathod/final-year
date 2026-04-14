from modules.llm import call_llm
import json


def select_architecture(system_analysis):

    prompt = f"""
You are a senior software architect.

Based on the system analysis below, choose the MOST appropriate architecture.

System Analysis:
{system_analysis}

Rules:
- DO NOT choose microservices for low or medium complexity systems
- Prefer layered or monolithic architecture for such systems
- Microservices ONLY if system is highly scalable, distributed, or very complex

Tasks:
1. Select architecture style
2. Select design pattern
3. Justify your choice

Return ONLY JSON:

{{
  "architecture_style": "...",
  "design_pattern": "...",
  "justification": "..."
}}
"""

    response = call_llm(prompt, model="mistral", temperature=0.2)

    try:
        data = json.loads(response)
    except:
        print("⚠️ Failed to parse architecture JSON")
        print(response)
        return {}

    return data