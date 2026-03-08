from modules.extractor import extract_requirements

class ExtractionAgent:

    def run(self, user_input):
        return extract_requirements(user_input)