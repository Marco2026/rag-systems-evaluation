import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from datasets import load_dataset, Dataset
from RAG.Rag import Rag
from collections import namedtuple
from pathlib import Path
from Config.settings import QUALITY_BENCHMARK_SYSTEM_PROMPT, CHUNK_SIZE, CHUNK_OVERLAP
from datetime import datetime
import json
import torch
import gc
import pandas as pd

Problem = namedtuple("Problem", "question, options, answer")

class ModelToEvaluate:

    def __init__(self, model_name, params, size):
        self.model_name = model_name
        self.params = params            # Medido en miles de millones de parámetros (B)
        self.size = size                # Medido en Gigabytes ocupados en memoria (GB)


class RAGEvaluator:

    def __init__(self, RAG, benchmark, retriever_to_evaluate, generator_to_evaluate):
        self.RAG = RAG
        self.benchmark = benchmark

        self.benchmark_start_time = None
        self.benchmark_finish_time = None

        self.retriever_to_evaluate = retriever_to_evaluate
        self.generator_to_evaluate = generator_to_evaluate

        output_dir = Path(f"Evaluation/results")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = output_dir / f"{benchmark}_{timestamp}.jsonl"


    def evaluate(self):
        match self.benchmark:
            case "QuaLITY_EASY":
                self.QuaLITY_EASY_evaluation()
            case "QuaLITY_HARD":
                self.QuaLITY_HARD_evaluation()
            case _:
                return
            
    
    def generate_report(self):
        results = []
        with open(self.results_file, "r") as f:
            for line in f:
                results.append(json.loads(line))

        total = len(results)
        correct = sum(1 for r in results if r["is_correct"])
        accuracy = correct / total if total > 0 else 0

        duration, total_seconds = extract_duration(start_datetime=self.benchmark_start_time, finish_datetime=self.benchmark_finish_time)

        report = {
            "benchmark": self.benchmark,
            "retriever_model": self.retriever_to_evaluate.model_name,
            "retriever_params": self.retriever_to_evaluate.params,
            "retriever_size": self.retriever_to_evaluate.size,
            "generator_model": self.generator_to_evaluate.model_name,
            "generator_params": self.generator_to_evaluate.params,
            "generator_size": self.generator_to_evaluate.size,
            "timestamp": datetime.now().isoformat(),
            "duration": duration,
            "duration (seconds)": total_seconds,
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "results": results
        }

        report_file = self.results_file.parent / f"{self.benchmark}_R:{sanitize_name(self.RAG.retriever_model_name)}_G:{sanitize_name(self.RAG.generator_model_name)}_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.results_file.unlink()

        print(f"\nScore: {correct}/{total} ({round(accuracy * 100, 2):.2%})")
        print(f"Report saved to: {report_file}")
        return report
            
    
    def QuaLITY_EASY_evaluation(self):
        def parse_answer(option):
            options = {
                'A': 0,
                'B': 1,
                'C': 2,
                'D': 3
            }
            if option.strip().rstrip(')') in options.keys():
                return options[option.strip().rstrip(')')]
            return 'Error'

        dataset = load_dataset("emozilla/quality", split="validation")
        df = dataset.to_pandas()
        easy_df = df[df["hard"] == False].reset_index(drop=True)
        easy_df = easy_df.sample(n=min(200, len(easy_df)), random_state=42).reset_index(drop=True)
        dataset_sample = Dataset.from_pandas(easy_df, preserve_index=False)

        articles: list[str] = list()
        metadata: list[str] = list()
        problems: list[Problem] = list()
        hits: int = 0
        
        for d in dataset_sample:
            chunks = chunk_text(d["article"], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            for c in chunks:
                articles.append(c)
                metadata.append({"source": d["article"][:15], "text": c})

            problems.append(Problem(d["question"], d["options"], d["answer"]))
        
        self.benchmark_start_time = datetime.now()

        self.RAG.build_rag(prepared_data=[articles, metadata])

        prob_idx = 0
        for p in problems:
            prob_idx += 1
            question = f"""Question: {p.question}

                A) {p.options[0]}
                B) {p.options[1]}
                C) {p.options[2]}
                D) {p.options[3]}"""

            rag_answer = self.RAG.prompt(query=question, benchmark=True)
            is_correct = parse_answer(rag_answer) == p.answer

            print("\nProblem #" + str(prob_idx))
            print("Correct answer: " + str(p.answer))
            print("Answer provided: " + str(parse_answer(rag_answer)))

            if is_correct:
                hits += 1
            else:
                print('RAG answer ' + rag_answer + ' did not match correct answer: ' + str(p.answer))

            with open(self.results_file, "a") as f:
                f.write(json.dumps({
                    "problem_index": prob_idx,
                    "question": p.question,
                    "options": p.options,
                    "correct_answer": p.answer,
                    "rag_answer": rag_answer,
                    "is_correct": is_correct
                }, ensure_ascii=False) + "\n")

            torch.cuda.empty_cache()
            gc.collect()

        self.benchmark_finish_time = datetime.now()

        self.generate_report()
        score = hits / len(problems)
        return score
    
    def QuaLITY_HARD_evaluation(self):
        pass


def sanitize_name(file_name):
    return file_name.replace('/', '-')


def extract_duration(start_datetime: datetime, finish_datetime: datetime) -> tuple[str, int]:
    duration = finish_datetime - start_datetime
    total_seconds = int(duration.total_seconds())
    hours   = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return (f"{hours:02}:{minutes:02}:{seconds:02}", total_seconds)


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(text), step):
        piece = text[i:i + chunk_size].strip()
        if piece:
            chunks.append(piece)
    return chunks


if __name__ == "__main__":

    #RETRIEVER_MODEL_NAME = "Octen/Octen-Embedding-0.6B"
    #GENERATOR_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct" #"Qwen/Qwen2.5-7B-Instruct"

    retriever_model_1 = ModelToEvaluate(
        model_name = 'embeddinggemma:latest',
        params = 0.118,
        size = 0.476
    )

    generator_model_1 = ModelToEvaluate(
        model_name = 'phi4-mini:3.8b',
        params = 3.84,
        size = 2.5
    )

    # QuaLITY evaluation
    print("-"*20 + " STARTING QuaLITY EVALUATION " + "-"*20)
    print("Models to evaluate: ")
    print("- Retriever: " + retriever_model_1.model_name)
    print("- Generator: " + generator_model_1.model_name)
    print()

    rag = Rag(
        retriever_model_name = retriever_model_1.model_name,
        retriever_model_mode = "api",
        generator_model_name = generator_model_1.model_name,
        generator_model_mode = "api",
        rebuild_index = True,
        system_prompt=QUALITY_BENCHMARK_SYSTEM_PROMPT
    )

    evaluator = RAGEvaluator(
        RAG=rag,
        benchmark="QuaLITY_EASY",
        retriever_to_evaluate=retriever_model_1,
        generator_to_evaluate=generator_model_1
    )

    evaluator.evaluate()