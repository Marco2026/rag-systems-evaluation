const benchmarkSelect = document.getElementById('benchmark-select');
const reloadBtn = document.getElementById('reload-btn');
const exportBtn = document.getElementById('export-btn');
const matrixContainer = document.getElementById('matrix-container');
const detailSummary = document.getElementById('detail-summary');
const detailResults = document.getElementById('detail-results');
const curveCaption = document.getElementById('curve-caption');
const accuracySourceToggle = document.getElementById('accuracy-source-toggle');

let currentBenchmark = null;
let currentMatrixData = null;
let selectedCell = null;
let currentSource = 'raw';

async function fetchJson(url) {
    const response = await fetch(url);
    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || 'Request failed');
    }
    return response.json();
}

function accuracyToColor(accuracy) {
    if (accuracy === null || accuracy === undefined) {
        return '#f2f2f2';
    }

    const clamped = Math.max(0, Math.min(1, accuracy));
    const red = Math.round(240 - clamped * 120);
    const green = Math.round(90 + clamped * 120);
    const blue = Math.round(90 + clamped * 30);
    return `rgb(${red}, ${green}, ${blue})`;
}

function renderTable(data) {
    currentMatrixData = data;
    const retrievers = data.retrievers;
    const generators = data.generators;
    const matrix = data.matrix;

    const table = document.createElement('table');
    table.className = 'matrix-table';

    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    const corner = document.createElement('th');
    corner.textContent = 'Retriever \\ Generator';
    headRow.appendChild(corner);

    generators.forEach((generator) => {
        const th = document.createElement('th');
        const btn = document.createElement('button');
        btn.className = 'top-header-btn';
        btn.textContent = generator;
        btn.title = 'Click para análisis con generator fijo';
        btn.addEventListener('click', () => loadCurve('col', generator));
        th.appendChild(btn);
        headRow.appendChild(th);
    });

    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');

    retrievers.forEach((retriever, rowIndex) => {
        const tr = document.createElement('tr');

        const sideTh = document.createElement('th');
        const sideBtn = document.createElement('button');
        sideBtn.className = 'side-header-btn';
        sideBtn.textContent = retriever;
        sideBtn.title = 'Click para análisis con retriever fijo';
        sideBtn.addEventListener('click', () => loadCurve('row', retriever));
        sideTh.appendChild(sideBtn);
        tr.appendChild(sideTh);

        generators.forEach((generator, colIndex) => {
            const td = document.createElement('td');
            const accuracy = matrix[rowIndex][colIndex];

            if (accuracy === null || accuracy === undefined) {
                td.textContent = '-';
            } else {
                const btn = document.createElement('button');
                btn.className = 'cell-btn';
                btn.style.backgroundColor = accuracyToColor(accuracy);
                btn.textContent = accuracy.toFixed(3);
                btn.dataset.retriever = retriever;
                btn.dataset.generator = generator;
                btn.addEventListener('click', () => selectCell(btn, retriever, generator));
                td.appendChild(btn);
            }
            tr.appendChild(td);
        });

        tbody.appendChild(tr);
    });

    table.appendChild(tbody);
    matrixContainer.innerHTML = '';
    matrixContainer.appendChild(table);
}

function renderDetail(detail) {
    const summary = detail.summary;
    const isPostSource = detail.source === 'post';

    detailSummary.innerHTML = `
        <div class="detail-list">
            <div class="detail-list-item"><span class="detail-label">Retriever</span><span class="detail-value">${detail.retriever}</span></div>
            <div class="detail-list-item"><span class="detail-label">Generator</span><span class="detail-value">${detail.generator}</span></div>
            <div class="detail-list-item"><span class="detail-label">Fuente</span><span class="detail-value">${isPostSource ? 'POST' : 'RAW'}</span></div>
            <div class="detail-list-item"><span class="detail-label">Accuracy</span><span class="detail-value">${summary.accuracy}</span></div>
            <div class="detail-list-item"><span class="detail-label">Correct</span><span class="detail-value">${summary.correct}/${summary.total}</span></div>
            <div class="detail-list-item"><span class="detail-label">Duracion</span><span class="detail-value">${summary.duration} (${summary.duration_seconds}s)</span></div>
        </div>
    `;

    const maxItems = 40;
    const items = detail.results.slice(0, maxItems);

    if (items.length === 0) {
        detailResults.innerHTML = '<div class="detail-item">No hay resultados por pregunta en este reporte.</div>';
        return;
    }

    const rows = items
        .map((result) => {
            const rowClass = result.is_correct ? 'correct' : 'incorrect';
            const postprocessedAnswer = result.parsed_option ?? '';
            const cleanAnswerCell = isPostSource ? `<td>${postprocessedAnswer}</td>` : '';

            return `
                <tr class="${rowClass}">
                    <td>${result.problem_index}</td>
                    <td>${result.correct_answer}</td>
                    <td>${result.rag_answer ?? ''}</td>
                    ${cleanAnswerCell}
                </tr>
            `;
        })
        .join('');

    const cleanAnswerHeader = isPostSource ? '<th>Respuesta postprocesada</th>' : '';

    detailResults.innerHTML = `
        <table class="detail-table">
            <thead>
                <tr>
                    <th>Numero de pregunta</th>
                    <th>Respuesta correcta</th>
                    <th>Respuesta del RAG</th>
                    ${cleanAnswerHeader}
                </tr>
            </thead>
            <tbody>
                ${rows}
            </tbody>
        </table>
    `;
}

function renderCurve(curve) {
    const points = curve.points;
    if (!points || points.length === 0) {
        curveCaption.textContent = 'No hay datos suficientes para esta selección.';
        Plotly.purge('curve-plot');
        return;
    }

    const x = points.map((point, idx) => (point.params_billions === null ? idx + 1 : point.params_billions));
    const y = points.map((point) => point.accuracy);
    const labels = points.map((point) => point.name);

    const trace = {
        type: 'scatter',
        mode: 'lines+markers+text',
        x,
        y,
        text: labels,
        textposition: 'top center',
        marker: { size: 8, color: '#e07d3c' },
        line: { color: '#1a1a1a', width: 2 },
    };

    const layout = {
        margin: { t: 18, r: 12, b: 52, l: 52 },
        yaxis: { range: [0, 1], title: 'Accuracy' },
        xaxis: { title: 'Parámetros (billions) o índice' },
    };

    curveCaption.textContent = `Análisis con ${curve.fixed.type} fijo: ${curve.fixed.name}`;
    Plotly.newPlot('curve-plot', [trace], layout, { responsive: true, displayModeBar: false });
}

async function selectCell(button, retriever, generator) {
    if (selectedCell) {
        selectedCell.classList.remove('active');
    }
    selectedCell = button;
    selectedCell.classList.add('active');

    const url = `/api/results/detail?benchmark=${encodeURIComponent(currentBenchmark)}&retriever=${encodeURIComponent(retriever)}&generator=${encodeURIComponent(generator)}&source=${encodeURIComponent(currentSource)}`;
    const detail = await fetchJson(url);
    renderDetail(detail);
}

async function loadCurve(axis, model) {
    if (!currentBenchmark) {
        return;
    }

    const url = `/api/results/curve?benchmark=${encodeURIComponent(currentBenchmark)}&axis=${encodeURIComponent(axis)}&model=${encodeURIComponent(model)}&source=${encodeURIComponent(currentSource)}`;
    const curve = await fetchJson(url);
    renderCurve(curve);
}

async function loadMatrix(benchmark) {
    currentBenchmark = benchmark;
    const matrix = await fetchJson(`/api/results/matrix?benchmark=${encodeURIComponent(benchmark)}&source=${encodeURIComponent(currentSource)}`);
    renderTable(matrix);

    detailSummary.textContent = 'Selecciona una celda para ver detalle.';
    detailResults.innerHTML = '';
    curveCaption.textContent = 'Selecciona una fila o columna para dibujar la curva.';
    Plotly.purge('curve-plot');
}

function setExportState(isLoading) {
    exportBtn.disabled = isLoading;
    exportBtn.textContent = isLoading ? 'Exportando...' : 'Exportar';
}

async function exportBenchmark() {
    if (!currentBenchmark) {
        return;
    }

    try {
        setExportState(true);
        const response = await fetch(`/api/results/export?benchmark=${encodeURIComponent(currentBenchmark)}`);
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(errorText || 'No se pudo exportar');
        }

        const blob = await response.blob();
        const disposition = response.headers.get('Content-Disposition') || '';
        const filenameMatch = disposition.match(/filename=([^;]+)/i);
        const filename = filenameMatch
            ? filenameMatch[1].trim().replace(/^"|"$/g, '')
            : `${currentBenchmark}_export.zip`;

        const url = URL.createObjectURL(blob);
        const anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = filename;
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        URL.revokeObjectURL(url);
    } catch (error) {
        console.error(error);
        alert(`Error exportando: ${error.message}`);
    } finally {
        setExportState(false);
    }
}

function updateSourceToggleLabel() {
    const isPost = currentSource === 'post';
    accuracySourceToggle.classList.toggle('on', isPost);
    accuracySourceToggle.setAttribute('aria-pressed', isPost ? 'true' : 'false');
    accuracySourceToggle.title = isPost
        ? 'Postprocesado activado: usando accuracy de Evaluation/postprocessed'
        : 'Postprocesado desactivado: usando accuracy de Evaluation/results';
}

async function loadBenchmarks(preferredBenchmark) {
    const data = await fetchJson(`/api/results/benchmarks?source=${encodeURIComponent(currentSource)}`);
    const benchmarks = data.benchmarks || [];

    benchmarkSelect.innerHTML = '';
    benchmarks.forEach((benchmark) => {
        const option = document.createElement('option');
        option.value = benchmark;
        option.textContent = benchmark;
        benchmarkSelect.appendChild(option);
    });

    if (benchmarks.length === 0) {
        currentBenchmark = null;
        exportBtn.disabled = true;
        matrixContainer.innerHTML = '<p>No se encontraron reportes para la fuente seleccionada.</p>';
        detailSummary.textContent = 'Selecciona una celda para ver detalle.';
        detailResults.innerHTML = '';
        curveCaption.textContent = 'Selecciona una fila o columna para dibujar la curva.';
        Plotly.purge('heatmap');
        Plotly.purge('curve-plot');
        return null;
    }

    const selected = preferredBenchmark && benchmarks.includes(preferredBenchmark)
        ? preferredBenchmark
        : benchmarks[0];
    benchmarkSelect.value = selected;
    exportBtn.disabled = false;
    return selected;
}

async function init() {
    try {
        updateSourceToggleLabel();
        const benchmark = await loadBenchmarks(null);
        if (!benchmark) {
            return;
        }
        await loadMatrix(benchmark);
    } catch (error) {
        matrixContainer.innerHTML = `<p>Error cargando resultados: ${error.message}</p>`;
    }
}

benchmarkSelect.addEventListener('change', async (event) => {
    await loadMatrix(event.target.value);
});

reloadBtn.addEventListener('click', async () => {
    const benchmark = await loadBenchmarks(benchmarkSelect.value);
    if (benchmark) {
        await loadMatrix(benchmark);
    }
});

exportBtn.addEventListener('click', exportBenchmark);

accuracySourceToggle.addEventListener('click', async () => {
    currentSource = currentSource === 'raw' ? 'post' : 'raw';
    updateSourceToggleLabel();

    const benchmark = await loadBenchmarks(benchmarkSelect.value || currentBenchmark);
    if (benchmark) {
        await loadMatrix(benchmark);
    }
});

document.addEventListener('DOMContentLoaded', init);
