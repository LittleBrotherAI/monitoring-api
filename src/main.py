from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def hello():
    return {"message": "Hello world!"}


# example for you guys. replace honesty with your metric
@app.get("/api/monitor/honesty")
def monitor_honesty():
    return {"honesty": 0}
