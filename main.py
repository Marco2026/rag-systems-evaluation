from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from RAG.Rag import Rag

app = FastAPI()
rag = Rag(rebuild_index=True)

app.mount("/static", StaticFiles(directory="App/static"), name="static")
templates = Jinja2Templates(directory="App/templates")

class Message(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


@app.post("/api/create_chat")
async def create_chat(request: Request):
    return


@app.post("/api/send_message")
async def send_message(request: Request):
    data = await request.json()
    answer = rag.prompt(data["message"])
    return {"message": answer}