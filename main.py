from fastapi import FastAPI
import uvicorn
from evaluation import evaluation
from trend import trend
from typing import List

app = FastAPI()

@app.post("/ai_tags")
async def ai_tag(articles:List):
    tags = trend().get_tags()
    response = evaluation(tags, articles)
    return response