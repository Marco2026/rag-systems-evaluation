from fastapi import FastAPI, Request
from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from RAG.Rag import Rag
from Evaluation.DataVisualizer import DataVisualizer

app = FastAPI()
rag = None
visualizer = DataVisualizer()

app.mount("/static", StaticFiles(directory="App/static"), name="static")
templates = Jinja2Templates(directory="App/templates")

class Message(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={"request": request},
    )


@app.get("/results", response_class=HTMLResponse)
async def results_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="results.html",
        context={"request": request},
    )


@app.post("/api/create_chat")
async def create_chat(request: Request):
    return


@app.post("/api/start_rag")
async def start_rag():
    global rag
    if rag is None:
        rag = Rag(rebuild_index=True)
        return {"status": "started", "message": "RAG started"}
    return {"status": "already_started", "message": "RAG is already running"}


@app.post("/api/send_message")
async def send_message(request: Request):
    if rag is None:
        raise HTTPException(status_code=400, detail="RAG is not started. Press 'Start RAG' first.")

    data = await request.json()
    answer = rag.prompt(data["message"])
    return {"message": answer}


@app.get("/api/results/benchmarks")
async def get_benchmarks():
    return {"benchmarks": visualizer.list_benchmarks()}


@app.get("/api/results/matrix")
async def get_matrix(benchmark: str):
    if not benchmark:
        raise HTTPException(status_code=400, detail="benchmark is required")

    matrix = visualizer.get_matrix(benchmark)
    return matrix


@app.get("/api/results/detail")
async def get_result_detail(benchmark: str, retriever: str, generator: str):
    if not benchmark or not retriever or not generator:
        raise HTTPException(
            status_code=400,
            detail="benchmark, retriever and generator are required",
        )

    detail = visualizer.get_cell_detail(
        benchmark=benchmark,
        retriever=retriever,
        generator=generator,
    )
    if detail is None:
        raise HTTPException(status_code=404, detail="Combination not found")

    return detail


@app.get("/api/results/curve")
async def get_curve(benchmark: str, axis: str, model: str):
    if not benchmark or not axis or not model:
        raise HTTPException(status_code=400, detail="benchmark, axis and model are required")
    if axis not in {"row", "col"}:
        raise HTTPException(status_code=400, detail="axis must be 'row' or 'col'")

    curve = visualizer.get_curve(benchmark=benchmark, axis=axis, model=model)
    if curve is None:
        raise HTTPException(status_code=404, detail="Curve data not found")
    return curve