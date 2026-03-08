from modules.llm import call_llm


def generate_improvement_plan(scores, current_srs, user_input):

    prompt = f"""
You are a Requirements Engineering expert.

The following SRS was evaluated with these scores:

Completeness: {scores['completeness']}%
Correctness: {scores['correctness']}%
Consistency: {scores['consistency']}%
Clarity: {scores['clarity']}%
Structure Compliance: {scores['structure_compliance']}%
Redundancy: {scores['redundancy']}%

Your task:

Create a structured improvement plan to improve the SRS.

Rules:
- Focus on weaknesses in the metrics
- Suggest specific improvement actions
- Do NOT rewrite the SRS
- Only produce an improvement plan

User Requirements:
{user_input}

Current SRS:
{current_srs}

Return the plan as numbered steps.
"""

    return call_llm(prompt)