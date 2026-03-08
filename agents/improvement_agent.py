from modules.llm import call_llm


class ImprovementAgent:

    def run(self, srs, evaluation):

        prompt = f"""
You are an SRS refinement agent.

Current Evaluation Scores:
{evaluation}

Current SRS to improve:
{srs}

Rules:
- Do NOT invent new requirements.
- Preserve original meaning.
- Improve clarity and specificity.
- Fix structural weaknesses.
- Remove ambiguities.
- Maintain requirement IDs.

Return improved full SRS only.
"""

        return call_llm(prompt, temperature=0.3)