import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EvaluationRecord:
	benchmark: str
	retriever: str
	generator: str
	report_file: Path
	report: dict[str, Any]
	retriever_params: float | None = None
	generator_params: float | None = None
	retriever_size: float | None = None
	generator_size: float | None = None


class DataVisualizer:
	def __init__(self, results_dir: Path | None = None):
		self.results_dir = results_dir or Path("Evaluation/results")
		self.postprocessed_dir = Path("Evaluation/postprocessed")

	def list_benchmarks(self, source: str = "raw") -> list[str]:
		benchmarks: set[str] = set()
		for file_path in self._iter_report_files(source=source):
			try:
				with open(file_path, "r", encoding="utf-8") as handle:
					report = json.load(handle)
			except (OSError, json.JSONDecodeError):
				continue

			benchmark = report.get("benchmark")
			if isinstance(benchmark, str) and benchmark:
				benchmarks.add(benchmark)
		return sorted(benchmarks)

	def get_matrix(self, benchmark: str, source: str = "raw") -> dict[str, Any]:
		records = self._load_records(benchmark, source=source)
		retrievers = self._sort_models_by_report(records, axis="retriever")
		generators = self._sort_models_by_report(records, axis="generator")

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
					"retriever_params": record.retriever_params,
					"generator_params": record.generator_params,
					"retriever_size": record.retriever_size,
					"generator_size": record.generator_size,
				}

			matrix.append(row)

		return {
			"benchmark": benchmark,
			"source": source,
			"retrievers": retrievers,
			"generators": generators,
			"matrix": matrix,
			"cell_meta": cell_meta,
		}

	def get_cell_detail(
		self, benchmark: str, retriever: str, generator: str, source: str = "raw"
	) -> dict[str, Any] | None:
		records = self._load_records(benchmark, source=source)
		record = self._find_record(records, retriever, generator)
		if record is None:
			return None

		report = record.report
		return {
			"benchmark": benchmark,
			"source": source,
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

	def get_curve(self, benchmark: str, axis: str, model: str, source: str = "raw") -> dict[str, Any] | None:
		records = self._load_records(benchmark, source=source)

		if axis == "row":
			points = [
				self._record_to_curve_point(
					record,
					variable_name=record.generator,
					params_billions=record.generator_params,
				)
				for record in records
				if record.retriever == model
			]
			fixed_label = "retriever"
			variable_label = "generator"
		elif axis == "col":
			points = [
				self._record_to_curve_point(
					record,
					variable_name=record.retriever,
					params_billions=record.retriever_params,
				)
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
			"source": source,
			"axis": axis,
			"fixed": {"type": fixed_label, "name": model},
			"variable_type": variable_label,
			"points": points,
		}

	def _load_records(self, benchmark: str, source: str = "raw") -> list[EvaluationRecord]:
		records: list[EvaluationRecord] = []
		for file_path in self._iter_report_files(source=source, benchmark=benchmark):
			try:
				with open(file_path, "r", encoding="utf-8") as handle:
					report = json.load(handle)
			except (OSError, json.JSONDecodeError):
				continue

			report_benchmark = report.get("benchmark")
			if not isinstance(report_benchmark, str) or report_benchmark != benchmark:
				continue

			retriever_name = report.get("retriever_model")
			generator_name = report.get("generator_model")

			if not isinstance(retriever_name, str) or not retriever_name:
				continue

			if not isinstance(generator_name, str) or not generator_name:
				continue

			records.append(
				EvaluationRecord(
					benchmark=report_benchmark,
					retriever=retriever_name,
					generator=generator_name,
					report_file=file_path,
					report=report,
					retriever_params=self._safe_float(report.get("retriever_params")),
					generator_params=self._safe_float(report.get("generator_params")),
					retriever_size=self._safe_float(report.get("retriever_size")),
					generator_size=self._safe_float(report.get("generator_size")),
				)
			)
		return records

	def _iter_report_files(self, source: str, benchmark: str | None = None):
		if source == "post":
			if benchmark:
				yield from self.postprocessed_dir.glob(f"{benchmark}/*_report.json")
			else:
				yield from self.postprocessed_dir.glob("*/*_report.json")
			return

		yield from self.results_dir.glob("*_report.json")

	def _find_record(
		self, records: list[EvaluationRecord], retriever: str, generator: str
	) -> EvaluationRecord | None:
		for record in records:
			if record.retriever == retriever and record.generator == generator:
				return record
		return None

	def _record_to_curve_point(
		self, record: EvaluationRecord, variable_name: str, params_billions: float | None
	) -> dict[str, Any] | None:
		accuracy = self._safe_float(record.report.get("accuracy"))
		if accuracy is None:
			return None

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

	def _sort_models_by_report(
		self, records: list[EvaluationRecord], axis: str
	) -> list[str]:
		model_to_params: dict[str, float | None] = {}

		for record in records:
			if axis == "retriever":
				name = record.retriever
				params = record.retriever_params
			else:
				name = record.generator
				params = record.generator_params

			if name not in model_to_params or (model_to_params[name] is None and params is not None):
				model_to_params[name] = params

		return sorted(
			model_to_params,
			key=lambda name: (
				model_to_params[name] is None,
				model_to_params[name] if model_to_params[name] is not None else 0.0,
				name,
			),
		)
