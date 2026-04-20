import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from dal import ItemCreate, ItemDAL, ItemRead, get_client, get_collection


@asynccontextmanager
async def lifespan(app: FastAPI):
    mongo_url = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
    client = get_client(mongo_url)
    app.state.dal = ItemDAL(get_collection(client))
    yield
    client.close()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/items", response_model=list[ItemRead], status_code=200)
async def get_all_items():
    return await app.state.dal.get_all()


@app.get("/items/{item_id}", response_model=ItemRead, status_code=200)
async def get_item(item_id: str):
    item = await app.state.dal.get_one(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Item Not Found")
    return item


@app.post("/items", response_model=ItemRead, status_code=201)
async def create_item(item: ItemCreate):
    return await app.state.dal.create(item)


@app.put("/items/{item_id}", response_model=ItemRead, status_code=200)
async def update_item(item_id: str, item: ItemCreate):
    updated = await app.state.dal.update(item_id, item)
    if updated is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return updated


@app.delete("/items/{item_id}", status_code=200)
async def delete_item(item_id: str):
    deleted = await app.state.dal.delete(item_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"message": "Item deleted", "id": item_id}


# Static files must be mounted AFTER all API routes
app.mount("/", StaticFiles(directory="static", html=True), name="static")
