import json
import re
from pathlib import Path
from typing import Any


class ReportPostProcessor:
    def __init__(
        self,
        results_dir: Path | None = None,
        output_dir: Path | None = None,
    ):
        self.results_dir = results_dir or Path("Evaluation/results")
        self.output_dir = output_dir or Path("Evaluation/postprocessed")

    def postprocess_all_reports(self) -> dict[str, int]:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        scanned = 0
        generated = 0
        skipped = 0

        for source_path in self.results_dir.glob("*_report.json"):
            scanned += 1
            if self._postprocess_report_if_needed(source_path):
                generated += 1
            else:
                skipped += 1

        return {"scanned": scanned, "generated": generated, "skipped": skipped}

    def _postprocess_report_if_needed(self, source_path: Path) -> bool:
        report = self._load_report(source_path)
        if report is None:
            return False

        benchmark = report.get("benchmark")
        benchmark_name = benchmark if isinstance(benchmark, str) and benchmark else "UNKNOWN"
        target_path = self.output_dir / benchmark_name / source_path.name

        if target_path.exists() and target_path.stat().st_mtime >= source_path.stat().st_mtime:
            return False

        processed = self._postprocess_report(report)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as handle:
            json.dump(processed, handle, indent=2, ensure_ascii=False)

        return True

    def _load_report(self, source_path: Path) -> dict[str, Any] | None:
        try:
            with open(source_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None

        if not isinstance(data, dict):
            return None
        return data

    def _postprocess_report(self, report: dict[str, Any]) -> dict[str, Any]:
        results = report.get("results")
        if not isinstance(results, list):
            return report

        corrected = 0
        invalid = 0

        for row in results:
            if not isinstance(row, dict):
                continue

            raw_answer = row.get("rag_answer", "")
            normalized = self._normalize_answer(raw_answer)
            parsed_option = self._extract_choice(normalized)

            parsed_index: int | None = None
            parse_status = "invalid"
            if parsed_option is not None:
                parsed_index = self._choice_to_index(parsed_option)
                parse_status = "ok"

            correct_option, correct_index = self._parse_correct_answer(row.get("correct_answer"))
            is_correct_post = False
            if parsed_option is not None and correct_option is not None:
                is_correct_post = parsed_option == correct_option
            elif parsed_index is not None and correct_index is not None:
                is_correct_post = parsed_index == correct_index

            if is_correct_post:
                corrected += 1
            if parse_status == "invalid":
                invalid += 1

            row["rag_answer_clean"] = normalized
            row["parsed_option"] = parsed_option
            row["parsed_answer"] = parsed_index
            row["parse_status"] = parse_status
            row["is_correct"] = is_correct_post

        total = len(results)
        report["correct"] = corrected
        report["total"] = total
        report["accuracy"] = (corrected / total) if total else 0.0
        report["invalid_answers"] = invalid
        report["is_postprocessed"] = True

        return report

    def _normalize_answer(self, value: Any) -> str:
        if not isinstance(value, str):
            return ""

        answer = value.strip().upper()
        answer = re.sub(r"^(ASSISTANT|SYSTEM|USER)\s*[:\-]\s*", "", answer)
        answer = answer.strip(" \n\t\r'\"`.,;:()[]{}")
        return answer

    def _extract_choice(self, answer: str) -> str | None:
        if not answer:
            return None

        direct_match = re.fullmatch(r"\s*([ABCD])\s*\)?\s*", answer)
        if direct_match:
            return direct_match.group(1)

        tagged_match = re.search(r"\b(?:ANSWER|OPTION|RESPUESTA|OPCION)\s*[:\-]?\s*([ABCD])\b", answer)
        if tagged_match:
            return tagged_match.group(1)

        generic_match = re.search(r"\b([ABCD])\b", answer)
        if generic_match:
            return generic_match.group(1)

        return None

    def _choice_to_index(self, choice: str) -> int | None:
        choices = {"A": 0, "B": 1, "C": 2, "D": 3}
        return choices.get(choice)

    def _parse_correct_answer(self, value: Any) -> tuple[str | None, int | None]:
        if value is None:
            return (None, None)

        # Handles RACE_HARD style labels: "A", "B", "C", "D"
        if isinstance(value, str):
            candidate = value.strip().upper()
            if candidate in {"A", "B", "C", "D"}:
                return (candidate, self._choice_to_index(candidate))

        # Handles QuaLITY style numeric labels: 0, 1, 2, 3
        try:
            index = int(value)
        except (TypeError, ValueError):
            return (None, None)

        choices_by_index = {0: "A", 1: "B", 2: "C", 3: "D"}
        option = choices_by_index.get(index)
        if option is None:
            return (None, None)
        return (option, index)
