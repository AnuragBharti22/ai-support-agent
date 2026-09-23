from fastapi import FastAPI
from pydantic import BaseModel
from agent import ask

app = FastAPI(title="AI Support Agent")


class QueryRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"message": "AI Support Agent is running. POST to /ask with a question."}


@app.post("/ask")
def ask_question(request: QueryRequest):
    answer = ask(request.question)
    return {"question": request.question, "answer": answer}