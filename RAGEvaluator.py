import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from datasets import load_dataset, Dataset
from RAG.Rag import Rag
from collections import namedtuple
from pathlib import Path
from Config.settings import QUALITY_BENCHMARK_SYSTEM_PROMPT, NO_RETRIEVER_SYSTEM_PROMPT, CHUNK_SIZE, CHUNK_OVERLAP
from datetime import datetime
import json
import torch
import gc
import pandas as pd

Problem = namedtuple("Problem", "question, options, answer")

class ModelToEvaluate:

    def __init__(self, model_name: str, model_mode: str, params: float, size: float):
        self.model_name = model_name
        self.model_mode = model_mode    # 'api' or 'local'
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
        print("-"*20 + " STARTING " + self.benchmark + " EVALUATION " + "-"*20)
        print("Models to evaluate: ")
        print("- Retriever: " + self.retriever_to_evaluate.model_name)
        print("- Generator: " + self.generator_to_evaluate.model_name)
        print()
        match self.benchmark:
            case "QuaLITY_EASY":
                self.QuaLITY_EASY_evaluation()
            case "QuaLITY_HARD":
                self.QuaLITY_HARD_evaluation()
            case "RACE_HARD":
                self.RACE_HARD_evaluation()
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
        hard_df = df[df["hard"] == True].reset_index(drop=True)
        hard_df = hard_df.sample(n=min(200, len(hard_df)), random_state=42).reset_index(drop=True)
        dataset_sample = Dataset.from_pandas(hard_df, preserve_index=False)

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
    

    def RACE_HARD_evaluation(self):
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

        dataset = load_dataset("ehovy/race", "high", split="validation")
        df = dataset.to_pandas()
        df_sample = df.sample(n=min(200, len(df)), random_state=42).reset_index(drop=True)
        dataset_sample = Dataset.from_pandas(df_sample, preserve_index=False)

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


def grid_evaluation(retrievers: list[ModelToEvaluate], generators: list[ModelToEvaluate], benchmark: str):
    for g in generators:
        for r in retrievers:
            rag = Rag(
                retriever_model_name = r.model_name,
                retriever_model_mode = r.model_mode,
                generator_model_name = g.model_name,
                generator_model_mode = g.model_mode,
                rebuild_index = True,
                system_prompt=NO_RETRIEVER_SYSTEM_PROMPT
            )
            rag_evaluator = RAGEvaluator(
                RAG=rag,
                benchmark=benchmark,
                retriever_to_evaluate=r,
                generator_to_evaluate=g
            )
            rag_evaluator.evaluate()


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

    # SIN RETRIEVER

    mock_retriever_model = ModelToEvaluate(
        model_name = 'MockRetriever',
        model_mode = 'mock',
        params = 0.,
        size = 0.
    )

    # MODELOS LOCALES

    local_retriever_model_1 = ModelToEvaluate(
        model_name = 'Octen/Octen-Embedding-0.6B',
        model_mode = 'local',
        params = 0.6,
        size = 2.5
    )

    local_generator_model_1 = ModelToEvaluate(
        model_name = 'Qwen/Qwen2.5-3B-Instruct',
        model_mode = 'local',
        params = 2.5,
        size = 4.2
    )

    # MODELOS EN SERVIDOR
    
    retriever_model_1 = ModelToEvaluate(
        model_name = 'jeffh/intfloat-multilingual-e5-small:f32',
        model_mode = 'api',
        params = 0.118,
        size = 0.4
    )

    retriever_model_2 = ModelToEvaluate(
        model_name = 'embeddinggemma:latest',
        model_mode = 'api',
        params = 0.308,
        size = 0.6
    )

    retriever_model_3 = ModelToEvaluate(
        model_name = 'qwen3-embedding:0.6b',
        model_mode = 'api',
        params = 0.596,
        size = 0.6
    )

    retriever_model_4 = ModelToEvaluate(
        model_name = 'miti99/gte-qwen2:latest',
        model_mode = 'api',
        params = 1.8,
        size = 3.3
    )

    retriever_model_5 = ModelToEvaluate(
        model_name = 'qwen3-embedding:4b',
        model_mode = 'api',
        params = 4.02,
        size = 2.3
    )

    generator_model_1 = ModelToEvaluate(
        model_name = 'phi4-mini:3.8b',
        model_mode = 'api',
        params = 3.84,
        size = 2.3
    )

    generator_model_2 = ModelToEvaluate(
        model_name = 'goekdenizguelmez/JOSIEFIED-Qwen2.5:7b',
        model_mode = 'api',
        params = 7.62,
        size = 4.4
    )

    generator_model_3 = ModelToEvaluate(
        model_name = 'rjmalagon/lamarck-v0.7:14b-bf16',
        model_mode = 'api',
        params = 14.8,
        size = 27.5
    )

    generator_model_4 = ModelToEvaluate(
        model_name = 'huihui_ai/fluentlylm-prinum-abliterated:32b',
        model_mode = 'api',
        params = 32.8,
        size = 18.5
    )

    generator_model_5 = ModelToEvaluate(
        model_name = 'huihui_ai/qwen2.5-abliterate:72b',
        model_mode = 'api',
        params = 72.7,
        size = 44.2
    )


    # Evaluation config
    retrievers_to_evaluate = [
        mock_retriever_model
    ]

    generators_to_evaluate = [
        local_generator_model_1
    ]

    benchmark_to_evaluate = "RACE_HARD"

    grid_evaluation(
        retrievers=retrievers_to_evaluate,
        generators=generators_to_evaluate,
        benchmark=benchmark_to_evaluate
    )

