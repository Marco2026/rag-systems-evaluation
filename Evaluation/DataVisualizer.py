import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPORT_RE = re.compile(
	r"^(?P<benchmark>.+?)_R:(?P<retriever>.+?)_G:(?P<generator>.+?)_report\.json$"
)


@dataclass
class EvaluationRecord:
	benchmark: str
	retriever: str
	generator: str
	report_file: Path
	report: dict[str, Any]


class DataVisualizer:
	def __init__(self, results_dir: Path | None = None):
		self.results_dir = results_dir or Path("Evaluation/results")

	def list_benchmarks(self) -> list[str]:
		benchmarks: set[str] = set()
		for file_path in self.results_dir.glob("*_report.json"):
			parsed = self._parse_report_filename(file_path.name)
			if parsed is not None:
				benchmarks.add(parsed["benchmark"])
		return sorted(benchmarks)

	def get_matrix(self, benchmark: str) -> dict[str, Any]:
		records = self._load_records(benchmark)
		retrievers = self._sort_models_by_params({record.retriever for record in records})
		generators = self._sort_models_by_params({record.generator for record in records})

		matrix: list[list[float | None]] = []
		cell_meta: dict[str, dict[str, dict[str, Any]]] = {}

		for retriever in retrievers:
			row: list[float | None] = []
			cell_meta[retriever] = {}

			for generator in generators:
				record = self._find_record(records, retriever, generator)
				if record is None:
					row.append(None)
					continue

				accuracy = self._safe_float(record.report.get("accuracy"))
				row.append(accuracy)
				cell_meta[retriever][generator] = {
					"accuracy": accuracy,
					"total": record.report.get("total"),
					"correct": record.report.get("correct"),
					"duration_seconds": record.report.get("duration (seconds)"),
					"timestamp": record.report.get("timestamp"),
					"report_file": str(record.report_file),
				}

			matrix.append(row)

		return {
			"benchmark": benchmark,
			"retrievers": retrievers,
			"generators": generators,
			"matrix": matrix,
			"cell_meta": cell_meta,
		}

	def get_cell_detail(
		self, benchmark: str, retriever: str, generator: str
	) -> dict[str, Any] | None:
		records = self._load_records(benchmark)
		record = self._find_record(records, retriever, generator)
		if record is None:
			return None

		report = record.report
		return {
			"benchmark": benchmark,
			"retriever": retriever,
			"generator": generator,
			"summary": {
				"accuracy": self._safe_float(report.get("accuracy")),
				"total": report.get("total"),
				"correct": report.get("correct"),
				"duration": report.get("duration"),
				"duration_seconds": report.get("duration (seconds)"),
				"timestamp": report.get("timestamp"),
				"report_file": str(record.report_file),
			},
			"results": report.get("results", []),
		}

	def get_curve(self, benchmark: str, axis: str, model: str) -> dict[str, Any] | None:
		records = self._load_records(benchmark)

		if axis == "row":
			points = [
				self._record_to_curve_point(record, variable_name=record.generator)
				for record in records
				if record.retriever == model
			]
			fixed_label = "retriever"
			variable_label = "generator"
		elif axis == "col":
			points = [
				self._record_to_curve_point(record, variable_name=record.retriever)
				for record in records
				if record.generator == model
			]
			fixed_label = "generator"
			variable_label = "retriever"
		else:
			return None

		points = [point for point in points if point is not None]
		points.sort(
			key=lambda point: (
				point["params_billions"] is None,
				point["params_billions"] if point["params_billions"] is not None else 0.0,
				point["name"],
			)
		)

		return {
			"benchmark": benchmark,
			"axis": axis,
			"fixed": {"type": fixed_label, "name": model},
			"variable_type": variable_label,
			"points": points,
		}

	def _load_records(self, benchmark: str) -> list[EvaluationRecord]:
		records: list[EvaluationRecord] = []
		for file_path in self.results_dir.glob(f"{benchmark}_R:*_G:*_report.json"):
			parsed = self._parse_report_filename(file_path.name)
			if parsed is None:
				continue

			with open(file_path, "r", encoding="utf-8") as handle:
				report = json.load(handle)

			records.append(
				EvaluationRecord(
					benchmark=parsed["benchmark"],
					retriever=parsed["retriever"],
					generator=parsed["generator"],
					report_file=file_path,
					report=report,
				)
			)
		return records

	def _parse_report_filename(self, filename: str) -> dict[str, str] | None:
		match = REPORT_RE.match(filename)
		if match is None:
			return None
		return {
			"benchmark": match.group("benchmark"),
			"retriever": match.group("retriever"),
			"generator": match.group("generator"),
		}

	def _find_record(
		self, records: list[EvaluationRecord], retriever: str, generator: str
	) -> EvaluationRecord | None:
		for record in records:
			if record.retriever == retriever and record.generator == generator:
				return record
		return None

	def _record_to_curve_point(
		self, record: EvaluationRecord, variable_name: str
	) -> dict[str, Any] | None:
		accuracy = self._safe_float(record.report.get("accuracy"))
		if accuracy is None:
			return None

		params_billions = self._extract_params_billions(variable_name)
		return {
			"name": variable_name,
			"accuracy": accuracy,
			"params_billions": params_billions,
		}

	def _safe_float(self, value: Any) -> float | None:
		try:
			if value is None:
				return None
			return float(value)
		except (TypeError, ValueError):
			return None

	def _extract_params_billions(self, model_name: str) -> float | None:
		# Match sizes like 0.6B, 3B, 70B or 560M in model names.
		match = re.search(r"(\d+(?:\.\d+)?)\s*([BbMm])", model_name)
		if match is None:
			return None

		value = float(match.group(1))
		unit = match.group(2).lower()
		if unit == "b":
			return value
		if unit == "m":
			return value / 1000.0
		return None

	def _sort_models_by_params(self, model_names: set[str]) -> list[str]:
		return sorted(
			model_names,
			key=lambda name: (
				self._extract_params_billions(name) is None,
				self._extract_params_billions(name) if self._extract_params_billions(name) is not None else 0.0,
				name,
			),
		)
