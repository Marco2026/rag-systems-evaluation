const benchmarkSelect = document.getElementById('benchmark-select');
const reloadBtn = document.getElementById('reload-btn');
const matrixContainer = document.getElementById('matrix-container');
const detailSummary = document.getElementById('detail-summary');
const detailResults = document.getElementById('detail-results');
const curveCaption = document.getElementById('curve-caption');

let currentBenchmark = null;
let currentMatrixData = null;
let selectedCell = null;

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

function renderHeatmap(data) {
    const trace = {
        type: 'heatmap',
        z: data.matrix,
        x: data.generators,
        y: data.retrievers,
        zmin: 0,
        zmax: 1,
        colorscale: 'YlGnBu',
        hoverongaps: false,
    };

    const layout = {
        margin: { t: 10, r: 10, b: 110, l: 170 },
        xaxis: { tickangle: -25 },
        yaxis: { automargin: true },
    };

    Plotly.newPlot('heatmap', [trace], layout, { responsive: true, displayModeBar: false });
}

function renderDetail(detail) {
    const summary = detail.summary;
    detailSummary.innerHTML = [
        `<strong>Retriever:</strong> ${detail.retriever}`,
        `<strong>Generator:</strong> ${detail.generator}`,
        `<strong>Accuracy:</strong> ${summary.accuracy}`,
        `<strong>Correct:</strong> ${summary.correct}/${summary.total}`,
        `<strong>Duración:</strong> ${summary.duration} (${summary.duration_seconds}s)`,
    ].join(' | ');

    const maxItems = 40;
    const items = detail.results.slice(0, maxItems);

    if (items.length === 0) {
        detailResults.innerHTML = '<div class="detail-item">No hay resultados por pregunta en este reporte.</div>';
        return;
    }

    detailResults.innerHTML = items
        .map((result) => {
            const rowClass = result.is_correct ? 'correct' : 'incorrect';
            return `
                <div class="detail-item ${rowClass}">
                    <strong>#${result.problem_index}</strong>
                    | correct_answer: ${result.correct_answer}
                    | rag_answer: ${result.rag_answer}
                </div>
            `;
        })
        .join('');
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

    const url = `/api/results/detail?benchmark=${encodeURIComponent(currentBenchmark)}&retriever=${encodeURIComponent(retriever)}&generator=${encodeURIComponent(generator)}`;
    const detail = await fetchJson(url);
    renderDetail(detail);
}

async function loadCurve(axis, model) {
    if (!currentBenchmark) {
        return;
    }

    const url = `/api/results/curve?benchmark=${encodeURIComponent(currentBenchmark)}&axis=${encodeURIComponent(axis)}&model=${encodeURIComponent(model)}`;
    const curve = await fetchJson(url);
    renderCurve(curve);
}

async function loadMatrix(benchmark) {
    currentBenchmark = benchmark;
    const matrix = await fetchJson(`/api/results/matrix?benchmark=${encodeURIComponent(benchmark)}`);
    renderTable(matrix);
    renderHeatmap(matrix);

    detailSummary.textContent = 'Selecciona una celda para ver detalle.';
    detailResults.innerHTML = '';
    curveCaption.textContent = 'Selecciona una fila o columna para dibujar la curva.';
    Plotly.purge('curve-plot');
}

async function init() {
    try {
        const data = await fetchJson('/api/results/benchmarks');
        const benchmarks = data.benchmarks || [];

        benchmarkSelect.innerHTML = '';
        benchmarks.forEach((benchmark) => {
            const option = document.createElement('option');
            option.value = benchmark;
            option.textContent = benchmark;
            benchmarkSelect.appendChild(option);
        });

        if (benchmarks.length === 0) {
            matrixContainer.innerHTML = '<p>No se encontraron reportes de benchmark en Evaluation/results.</p>';
            return;
        }

        await loadMatrix(benchmarks[0]);
    } catch (error) {
        matrixContainer.innerHTML = `<p>Error cargando resultados: ${error.message}</p>`;
    }
}

benchmarkSelect.addEventListener('change', async (event) => {
    await loadMatrix(event.target.value);
});

reloadBtn.addEventListener('click', async () => {
    await loadMatrix(benchmarkSelect.value);
});

document.addEventListener('DOMContentLoaded', init);
