from fastapi import FastAPI, Request
from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import threading
from RAG.Rag import Rag
from Evaluation.DataVisualizer import DataVisualizer
from Config.settings import SYSTEM_PROMPT

app = FastAPI()
rag = None
rag_initializing = False
rag_init_error = None
rag_lock = threading.Lock()
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
    global rag, rag_initializing, rag_init_error

    def _initialize_rag_background():
        global rag, rag_initializing, rag_init_error
        try:
            rag_instance = Rag(
                retriever_model_name="Octen/Octen-Embedding-0.6B",
                retriever_model_mode="local",
                generator_model_name="Qwen/Qwen2.5-3B-Instruct",
                generator_model_mode="local",
                rebuild_index=True,
                system_prompt=SYSTEM_PROMPT,
            )
            # Build everything so "ready" means prompts can be answered immediately.
            rag_instance.build_rag(prepared_data=[])
            with rag_lock:
                rag = rag_instance
                rag_init_error = None
        except Exception as exc:
            with rag_lock:
                rag = None
                rag_init_error = str(exc)
        finally:
            with rag_lock:
                rag_initializing = False

    with rag_lock:
        if rag is not None:
            return {"status": "ready", "message": "RAG is already running"}
        if rag_initializing:
            return {"status": "initializing", "message": "RAG is initializing"}

        rag_initializing = True
        rag_init_error = None

    threading.Thread(target=_initialize_rag_background, daemon=True).start()
    return {"status": "initializing", "message": "RAG initialization started"}


@app.get("/api/rag_status")
async def rag_status():
    with rag_lock:
        if rag is not None:
            return {"status": "ready"}
        if rag_initializing:
            return {"status": "initializing"}
        if rag_init_error:
            return {"status": "error", "message": rag_init_error}
        return {"status": "not_started"}


@app.post("/api/send_message")
async def send_message(request: Request):
    with rag_lock:
        rag_instance = rag
        initializing = rag_initializing

    if initializing:
        raise HTTPException(status_code=409, detail="RAG is still initializing. Try again in a few seconds.")
    if rag_instance is None:
        raise HTTPException(status_code=400, detail="RAG is not started. Press 'Start RAG' first.")

    data = await request.json()
    answer = rag_instance.prompt(query=data["message"], benchmark=False)
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