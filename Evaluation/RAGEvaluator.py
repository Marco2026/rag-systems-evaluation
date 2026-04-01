from datasets import load_dataset
from RAG.Rag import Rag
from collections import namedtuple
from Config.settings import QUALITY_BENCHMARK_SYSTEM_PROMPT

Problem = namedtuple("Problem", "question, options, answer")

class RAGEvaluator:

    def __init__(self, RAG, benchmark):
        self.RAG = RAG
        self.benchmark = benchmark


    def evaluate(self):
        match self.benchmark:
            case "QuaLITY":
                self.QuaLITY_evaluation()
            case _:
                return
            
    
    def QuaLITY_evaluation(self):
        def parse_answer(option):
            options = {
                'A': 0,
                'B': 1,
                'C': 2,
                'D': 3
            }
            return options[option]

        dataset = load_dataset("emozilla/quality", split="validation")

        articles = list[str]
        problems = list[Problem]
        hits = int
        
        for d in dataset:
            articles.append(d["article"])
            problems.append(Problem(d["question"], d["options"], d["answer"]))
        
        self.RAG.build_rag(chunked_data=articles)

        for p in problems:
            question = f"""Question: {p["question"]}

                A) {p["options"][0]}
                B) {p["options"][1]}
                C) {p["options"][2]}
                D) {p["options"][3]}"""

            rag_answer = rag.prompt(query=question, benchmark=True)
            if parse_answer(rag_answer) == p["answer"]: 
                hits += 1
        
        score = hits / len(problems)
        return score



if __name__ == "__main__":

    RETRIEVER_MODEL_NAME = "Octen/Octen-Embedding-0.6B"
    GENERATOR_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

    # QuaLITY evaluation
    print("-"*20 + "STARTING QuaLITY EVALUATION " + "-"*20)
    print("Models to evaluate: ")
    print("- Retriever: " + RETRIEVER_MODEL_NAME)
    print("- Generator: " + GENERATOR_MODEL_NAME)
    print()

    rag = Rag(
        retriever_model_name = RETRIEVER_MODEL_NAME,
        retriever_model_mode = "local",
        generator_model_name = GENERATOR_MODEL_NAME,
        generator_model_mode = "local",
        rebuild_index = True,
        system_prompt=QUALITY_BENCHMARK_SYSTEM_PROMPT
    )

    evaluator = RAGEvaluator(RAG=rag, benchmark="QuaLITY")
    evaluator.evaluate()