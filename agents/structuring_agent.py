from modules.generator import generate_srs
from modules.retriever import retrieve

class StructuringAgent:

    def __init__(self, index=None, chunks=None):
        self.index = index
        self.chunks = chunks

    def run(self, requirements, user_input):

        rag_context = None
        if self.index:
            rag_context = retrieve(user_input, self.index, self.chunks)

        return generate_srs(requirements, rag_context)