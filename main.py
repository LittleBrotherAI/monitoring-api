from adversarialLLM import call_judge_adversarial_llm
from factcheckLLM import call_judge_factcheck_llm
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def hello():
    return {"message": "Hello world!"}

@app.get("/api/monitor/factcheck")
def handle_factcheck():
    return call_judge_factcheck_llm("fake prompt", "fake cot", "fake answer")

@app.get("/api/monitor/adversarial")
def handle_adversarial():
    return call_judge_adversarial_llm("fake prompt", "fake cot", "fake answer")