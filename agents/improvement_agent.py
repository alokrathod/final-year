from modules.llm import call_llm


class ImprovementAgent:

    def run(self, srs, evaluation):

        prompt = f"""
You are an SRS refinement agent.

Current Evaluation Scores:
{evaluation}

Current SRS:
{srs}

STRICT RULES:
- DO NOT add new requirements
- DO NOT create new requirement IDs
- DO NOT introduce new features, roles, or functionality
- ONLY improve wording, clarity, and measurability

FOCUS ON:
- Making requirements TESTABLE (add measurable criteria)
- Improving correctness (align strictly with user input)
- Reducing ambiguity
- Fixing weak metrics ONLY

IMPORTANT:
- Preserve ALL original requirement IDs
- Keep requirements atomic
- Do NOT expand scope

Return improved full SRS only.
"""

        return call_llm(prompt, temperature=0.3)