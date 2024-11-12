from fastapi import FastAPI
from app.api import endpoints

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "GraphRAG FastAPI is running!"}

app.include_router(endpoints.router)
