from adversarialLLM import call_judge_adversarial_llm
from factcheckLLM import call_judge_factcheck_llm
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class LLMResponse(BaseModel):
    user_request: str
    model_cot: str
    model_answer: str

@app.get("/")
def hello():
    return {"message": "Hello world!"}

@app.get("/api/monitor/factcheck")
def handle_factcheck(llm_response: LLMResponse):
    return call_judge_factcheck_llm(llm_response.user_request, llm_response.model_cot, llm_response.model_answer)

@app.get("/api/monitor/adversarial")
def handle_adversarial(llm_response: LLMResponse):
    return call_judge_adversarial_llm(llm_response.user_request, llm_response.model_cot, llm_response.model_answer)