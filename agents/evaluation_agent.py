from modules.evaluator import evaluate_srs

class EvaluationAgent:

    def run(self, user_input, srs):
        return evaluate_srs(user_input, srs)