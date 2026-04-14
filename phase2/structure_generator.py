from modules.llm import call_llm


def generate_structure(architecture, tech_stack, analysis):

    prompt = f"""
You are a software architect.

Generate a clean and minimal MERN-based project structure.

Architecture:
{architecture}

System Features:
{analysis["features"]}

Tech stack:
{tech_stack}

STRICT GUIDELINES:
- Structure must be based on the given system features
- Identify key functional domains from the features and reflect them in the backend structure
- Keep the structure minimal and relevant to the features
- Let the internal organization (files/folders) be decided naturally based on best practices

- Do NOT:
  - hardcode specific entities
  - add unnecessary folders
  - include tools, configs, or boilerplate unless essential
  - explain anything

Output ONLY a clean folder tree
"""

    response = call_llm(prompt, model="mistral", temperature=0.2)

    return response