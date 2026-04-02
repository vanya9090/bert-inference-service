from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import asyncio
from typing import List

model = None
tokenizer = None
device = torch.device("cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    model_name = "sergeyzh/rubert-mini-frida"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    yield
    model = None


app = FastAPI(lifespan=lifespan)


class EmbedRequest(BaseModel):
    text: str


class EmbedResponse(BaseModel):
    embedding: list[float]


def compute_embedding(text: str) -> List[float]:
    assert tokenizer is not None
    assert model is not None

    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().tolist()
    return embedding


@app.post("/embed", response_model=EmbedResponse)
async def embed_text(req: EmbedRequest):
    try:
        embedding = await asyncio.to_thread(compute_embedding, req.text)
        return EmbedResponse(embedding=embedding)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    if model is not None:
        return {"status": "ok"}
    raise HTTPException(status_code=503, detail="Model not loaded")
