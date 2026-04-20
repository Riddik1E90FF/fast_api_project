from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI()

# Pydantic Model

class Item(BaseModel):
    name: str
    description: str | None = None 
    


# Endpoints

# Get all items
@app.get("/items", status_code=200)
def get_all_items():
    return list(items_db.values())

# Get individual item with item_id 
@app.get("/items/{item_id}", status_code=200)
def get_item(item_id: int):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item Not Found")
    return items_db[item_id]


# Create item
@app.post("/items", status_code=201)
def create_item(item: Item):
    global next_id
    new_item = {"id": next_id, "name": item.name, "description": item.description}
    items_db[next_id] = new_item
    next_id += 1
    return new_item

# Update item
@app.put("/items/{item_id}", status_code=200)
def update_item(item_id: int, item: Item):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    items_db[item_id].update({"name": item.name, "description": item.description})
    return items_db[item_id]

# Delete item
@app.delete("/items/{item_id}", status_code=200)
def delete_item(item_id: int):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    deleted = items_db.pop(item_id)
    return {"message": "Item deleted", "item": deleted}