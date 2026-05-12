import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


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
	_LATEX_REPLACEMENTS = {
		"\\": r"\\textbackslash{}",
		"&": r"\\&",
		"%": r"\\%",
		"#": r"\\#",
		"_": r"\\_",
		"{": r"\\{",
		"}": r"\\}",
		"~": r"\\textasciitilde{}",
		"^": r"\\textasciicircum{}",
	}

	def __init__(self, results_dir: Path | None = None):
		self.results_dir = results_dir or Path("Evaluation/results")
		self.postprocessed_dir = Path("Evaluation/postprocessed")

	def list_benchmarks(self, source: str = "raw") -> list[str]:
		benchmarks: set[str] = set()
		for file_path in self._iter_report_files(source=source):
			report = self._load_json(file_path)
			if report is None:
				continue

			benchmark = report.get("benchmark")
			if isinstance(benchmark, str) and benchmark:
				benchmarks.add(benchmark)
		return sorted(benchmarks)

	def get_matrix(self, benchmark: str, source: str = "raw") -> dict[str, Any]:
		records = self._load_records(benchmark, source=source)
		retrievers, retriever_params_map = self._collect_model_info(records, axis="retriever")
		generators, generator_params_map = self._collect_model_info(records, axis="generator")

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
				cell_meta[retriever][generator] = self._build_cell_meta(record, accuracy)

			matrix.append(row)

		return {
			"benchmark": benchmark,
			"source": source,
			"retrievers": retrievers,
			"generators": generators,
			"matrix": matrix,
			"cell_meta": cell_meta,
			"retriever_params_map": retriever_params_map,
			"generator_params_map": generator_params_map,
		}

	def get_cell_detail(
		self, benchmark: str, retriever: str, generator: str, source: str = "raw"
	) -> dict[str, Any] | None:
		records = self._load_records(benchmark, source=source)
		record = self._find_record(records, retriever, generator)
		if record is None:
			return None

		return {
			"benchmark": benchmark,
			"source": source,
			"retriever": retriever,
			"generator": generator,
			"summary": self._build_summary(record),
			"results": record.report.get("results", []),
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
			report = self._load_json(file_path)
			if report is None:
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

	def _collect_model_info(
		self, records: list[EvaluationRecord], axis: str
	) -> tuple[list[str], dict[str, float | None]]:
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

		sorted_names = sorted(
			model_to_params,
			key=lambda name: (
				model_to_params[name] is None,
				model_to_params[name] if model_to_params[name] is not None else 0.0,
				name,
			),
		)

		return sorted_names, model_to_params

	def build_latex_table(self, benchmark: str, source: str, source_label: str) -> str:
		payload = self.get_matrix(benchmark=benchmark, source=source)
		return self._build_latex_table_from_payload(payload, benchmark, source_label)

	def export_benchmark_zip(
		self,
		benchmark: str,
		postprocess: Callable[[], None] | None = None,
	) -> tuple[io.BytesIO, str]:
		if postprocess is not None:
			postprocess()

		raw_payload = self.get_matrix(benchmark=benchmark, source="raw")
		post_payload = self.get_matrix(benchmark=benchmark, source="post")

		if not raw_payload.get("retrievers") and not post_payload.get("retrievers"):
			raise ValueError("No data found for benchmark")

		safe_benchmark = self._label_safe(benchmark)
		zip_buffer = io.BytesIO()

		with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
			for payload, source_label in ((raw_payload, "raw"), (post_payload, "post")):
				self._write_payload_assets(
					zip_file,
					payload,
					benchmark,
					source_label,
					f"{safe_benchmark}_{source_label}",
				)

		zip_buffer.seek(0)
		return zip_buffer, f"{safe_benchmark}_export.zip"

	def export_all_benchmarks_zip(
		self,
		postprocess: Callable[[], None] | None = None,
	) -> tuple[io.BytesIO, str]:
		if postprocess is not None:
			postprocess()

		benchmarks = self._list_all_benchmarks()
		if not benchmarks:
			raise ValueError("No data found for benchmarks")

		zip_buffer = io.BytesIO()
		with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
			for benchmark in benchmarks:
				safe_benchmark = self._label_safe(benchmark)
				for source_label in ("raw", "post"):
					payload = self.get_matrix(benchmark=benchmark, source=source_label)
					if not payload.get("retrievers"):
						continue
					self._write_payload_assets(
						zip_file,
						payload,
						benchmark,
						source_label,
						f"{safe_benchmark}/{safe_benchmark}_{source_label}",
					)

			for source_label in ("raw", "post"):
				aggregate = self._aggregate_matrix(benchmarks, source_label)
				retrievers = aggregate["retrievers"]
				generators = aggregate["generators"]
				matrix = aggregate["matrix"]
				retriever_params = aggregate["retriever_params_map"]
				generator_params = aggregate["generator_params_map"]

				if retrievers and generators and matrix:
					self._write_plot_pair(
						zip_file,
						aggregate,
						source_label,
						"global/global",
						"media",
					)

					global_payload = {
						"retrievers": retrievers,
						"generators": generators,
						"matrix": matrix,
						"retriever_params_map": retriever_params,
						"generator_params_map": generator_params,
					}
					global_table = self._build_latex_table_from_payload(
						global_payload,
						"GLOBAL",
						source_label.upper(),
					)
					zip_file.writestr(
						f"global/global_{source_label}.md",
						global_table + "\n",
					)

		zip_buffer.seek(0)
		return zip_buffer, "benchmarks_export.zip"

	def _build_latex_table_from_payload(
		self,
		payload: dict[str, Any],
		benchmark: str,
		source_label: str,
	) -> str:
		retrievers = payload.get("retrievers", [])
		generators = payload.get("generators", [])
		retriever_params = payload.get("retriever_params_map", {})
		generator_params = payload.get("generator_params_map", {})
		matrix = payload.get("matrix", [])

		display_retrievers = [self._display_retriever_label(name) for name in retrievers]
		sin_index = next(
			(idx for idx, label in enumerate(display_retrievers) if label == "Sin Retriever"),
			None,
		)
		row_order = list(range(len(retrievers)))
		if sin_index is not None:
			row_order = [sin_index] + [idx for idx in row_order if idx != sin_index]

		column_spec = "l" + ("c" * len(generators))
		display_benchmark = benchmark.replace("\\", "")
		caption = (
			"Resultados de evaluación por combinación de modelos para el benchmark "
			f"{display_benchmark} ({self._format_source_label(source_label)})"
		)
		label = f"tab:resultados_benchmark_{self._label_safe(benchmark)}_{source_label}"

		lines = [
			"\\begin{table}",
			"\\centering",
			f"\\caption{{{self._latex_escape(caption)}}}",
			f"\\label{{{label}}}",
			f"\\begin{{tabular}}{{{column_spec}}}",
			"\\toprule",
		]

		generator_labels = []
		for idx, generator in enumerate(generators, start=1):
			params_label = self._format_params(generator_params.get(generator))
			generator_labels.append(f"G{idx} ({params_label})")
		header_cells = ["\\textbf{Retriever / Generator}"]
		header_cells.extend([f"\\textbf{{{label}}}" for label in generator_labels])
		lines.append(" & ".join(header_cells) + " \\\\")
		lines.append("\\midrule")

		retriever_labels: dict[int, str] = {}
		counter = 1
		for row_idx in row_order:
			if sin_index is not None and row_idx == sin_index:
				retriever_labels[row_idx] = "Sin Retriever"
			else:
				params_label = self._format_params(retriever_params.get(retrievers[row_idx]))
				retriever_labels[row_idx] = f"R{counter} ({params_label})"
				counter += 1

		for idx, row_idx in enumerate(row_order):
			retriever_label = retriever_labels.get(row_idx, "R?")
			row = matrix[row_idx] if row_idx < len(matrix) else []
			values = []
			for col_idx in range(len(generators)):
				cell = row[col_idx] if col_idx < len(row) else None
				if cell is None:
					values.append("0")
				else:
					values.append(f"{cell:.3f}")

			row_cells = [
				self._latex_escape(retriever_label),
				*values,
			]

			suffix = " \\\\[4pt]" if idx < len(row_order) - 1 else " \\\\"
			lines.append(" & ".join(row_cells) + suffix)

		col_values: list[list[float]] = [[] for _ in generators]
		for row in matrix:
			for col_idx in range(len(generators)):
				value = row[col_idx] if col_idx < len(row) else None
				if value is not None:
					col_values[col_idx].append(value)

		mean_row = ["Media"]
		std_row = ["Desviación típica"]
		for values in col_values:
			mean_row.append(f"{_mean(values):.3f}" if values else "0")
			std_row.append(f"{_std(values):.3f}" if values else "0")

		lines.append("\\midrule")
		lines.append(" & ".join(mean_row) + " \\\\")
		lines.append(" & ".join(std_row) + " \\\\")

		lines.extend(
			[
				"\\bottomrule",
				"\\end{tabular}",
				"\\end{table}",
			]
		)

		return "\n".join(lines)

	def _build_axis_values(
		self,
		models: list[str],
		params_map: dict[str, float | None],
	) -> list[float]:
		axis_values: list[float] = []
		for idx, name in enumerate(models, start=1):
			value = params_map.get(name)
			axis_values.append(float(value) if value is not None else float(idx))
		return axis_values

	def _build_summary(self, record: EvaluationRecord) -> dict[str, Any]:
		report = record.report
		return {
			"accuracy": self._safe_float(report.get("accuracy")),
			"total": report.get("total"),
			"correct": report.get("correct"),
			"duration": report.get("duration"),
			"duration_seconds": report.get("duration (seconds)"),
			"timestamp": report.get("timestamp"),
			"report_file": str(record.report_file),
		}

	def _build_cell_meta(
		self, record: EvaluationRecord, accuracy: float | None
	) -> dict[str, Any]:
		return {
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

	def _list_all_benchmarks(self) -> list[str]:
		raw_benchmarks = self.list_benchmarks(source="raw")
		post_benchmarks = self.list_benchmarks(source="post")
		return sorted(set(raw_benchmarks) | set(post_benchmarks))

	def _load_records_for_benchmarks(
		self,
		benchmarks: list[str],
		source: str,
	) -> list[EvaluationRecord]:
		records: list[EvaluationRecord] = []
		for benchmark in benchmarks:
			records.extend(self._load_records(benchmark, source=source))
		return records

	def _aggregate_matrix(
		self,
		benchmarks: list[str],
		source: str,
	) -> dict[str, Any]:
		records = self._load_records_for_benchmarks(benchmarks, source)
		retrievers, retriever_params_map = self._collect_model_info(records, axis="retriever")
		generators, generator_params_map = self._collect_model_info(records, axis="generator")

		pair_scores: dict[tuple[str, str], list[float]] = {}

		for record in records:
			accuracy = self._safe_float(record.report.get("accuracy"))
			if accuracy is None:
				continue
			pair_scores.setdefault((record.retriever, record.generator), []).append(accuracy)

		matrix: list[list[float | None]] = []
		for retriever in retrievers:
			row: list[float | None] = []
			for generator in generators:
				values = pair_scores.get((retriever, generator))
				if not values:
					row.append(None)
				else:
					row.append(sum(values) / len(values))
			matrix.append(row)

		return {
			"retrievers": retrievers,
			"generators": generators,
			"matrix": matrix,
			"retriever_params_map": retriever_params_map,
			"generator_params_map": generator_params_map,
		}

	def _plot_accuracy_series(
		self,
		output: io.BytesIO,
		title: str,
		x_values: list[float],
		series: list[tuple[str, list[float | None]]],
		x_label: str,
		y_label: str,
	) -> None:
		import matplotlib

		matplotlib.use("Agg")
		import matplotlib.pyplot as plt

		fig, ax = plt.subplots(figsize=(8.2, 4.6))
		for name, y_values in series:
			ax.plot(x_values, y_values, marker="o", linewidth=1.8, label=name)

		ax.set_title(title)
		ax.set_xlabel(x_label)
		ax.set_ylabel(y_label)
		ax.set_ylim(0, 1)
		ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
		ax.legend(fontsize="small", loc="best")
		fig.tight_layout()
		fig.savefig(output, format="png", dpi=180)
		plt.close(fig)

	def display_retriever_label(self, name: str) -> str:
		return self._display_retriever_label(name)

	def _display_retriever_label(self, name: str) -> str:
		lowered = name.strip().lower()
		if lowered in {"sin retriever", "no retriever", "none"}:
			return "Sin Retriever"
		if "mockretriever" in lowered or "mock retriever" in lowered:
			return "Sin Retriever"
		return name

	def _latex_escape(self, text: str) -> str:
		return "".join(self._LATEX_REPLACEMENTS.get(char, char) for char in text)

	def _load_json(self, file_path: Path) -> dict[str, Any] | None:
		try:
			with open(file_path, "r", encoding="utf-8") as handle:
				return json.load(handle)
		except (OSError, json.JSONDecodeError):
			return None

	def _write_payload_assets(
		self,
		zip_file: zipfile.ZipFile,
		payload: dict[str, Any],
		benchmark: str,
		source_label: str,
		file_prefix: str,
	) -> None:
		latex_table = self._build_latex_table_from_payload(
			payload,
			benchmark,
			source_label.upper(),
		)
		zip_file.writestr(f"{file_prefix}.md", latex_table + "\n")
		self._write_plot_pair(zip_file, payload, source_label, file_prefix, "")

	def _write_plot_pair(
		self,
		zip_file: zipfile.ZipFile,
		payload: dict[str, Any],
		source_label: str,
		file_prefix: str,
		title_prefix: str,
	) -> None:
		retrievers = payload.get("retrievers", [])
		generators = payload.get("generators", [])
		matrix = payload.get("matrix", [])
		retriever_params = payload.get("retriever_params_map", {})
		generator_params = payload.get("generator_params_map", {})

		if not (retrievers and generators and matrix):
			return

		metric_label = "" if not title_prefix else f"{title_prefix} "
		x_generators = self._build_axis_values(generators, generator_params)
		retriever_series: list[tuple[str, list[float | None]]] = []
		for row_idx, retriever in enumerate(retrievers):
			y_values = matrix[row_idx] if row_idx < len(matrix) else []
			retriever_series.append((self._display_retriever_label(retriever), y_values))

		retriever_plot = io.BytesIO()
		self._plot_accuracy_series(
			retriever_plot,
			f"Accuracy {metric_label}por retriever ({self._format_source_label(source_label)})",
			x_generators,
			retriever_series,
			"Parámetros de generadores (billions)",
			"Accuracy",
		)
		zip_file.writestr(f"{file_prefix}_retrievers.png", retriever_plot.getvalue())

		x_retrievers = self._build_axis_values(retrievers, retriever_params)
		generator_series: list[tuple[str, list[float | None]]] = []
		for col_idx, generator in enumerate(generators):
			column = []
			for row in matrix:
				column.append(row[col_idx] if col_idx < len(row) else None)
			generator_series.append((generator, column))

		generator_plot = io.BytesIO()
		self._plot_accuracy_series(
			generator_plot,
			f"Accuracy {metric_label}por generator ({self._format_source_label(source_label)})",
			x_retrievers,
			generator_series,
			"Parámetros de retrievers (billions)",
			"Accuracy",
		)
		zip_file.writestr(f"{file_prefix}_generators.png", generator_plot.getvalue())

	def _format_params(self, value: float | None) -> str:
		if value is None:
			return "?"
		return f"{value:.2f}B"

	def _label_safe(self, text: str) -> str:
		return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_") or "benchmark"
	
	def _format_source_label(self, source_label: str) -> str:
	    return "con postprocesamiento" if source_label.lower() == "post" else "sin postprocesamiento"
	
def _mean(values: list[float]) -> float:
	return sum(values) / len(values)

def _std(values: list[float]) -> float:
	if len(values) <= 1:
		return 0.0
	mean = _mean(values)
	var = sum((v - mean) ** 2 for v in values) / len(values)
	return var ** 0.5
