from typing import Optional

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ItemCreate(BaseModel):
    """Fields the caller provides when creating or updating an item."""
    name: str
    description: Optional[str] = None


class ItemRead(BaseModel):
    """What the API returns — ObjectId serialised as a plain string 'id'."""
    id: str = Field(alias="_id")
    name: str
    description: Optional[str] = None

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc_to_item(doc: dict) -> ItemRead:
    """Convert a raw MongoDB document to ItemRead, stringifying the ObjectId."""
    doc["_id"] = str(doc["_id"])
    return ItemRead(**doc)


def get_client(mongo_url: str) -> AsyncIOMotorClient:
    return AsyncIOMotorClient(mongo_url)


def get_collection(client: AsyncIOMotorClient):
    return client["items_db"]["items"]


# ---------------------------------------------------------------------------
# Data Access Layer
# ---------------------------------------------------------------------------

class ItemDAL:
    def __init__(self, collection):
        self._collection = collection

    async def get_all(self) -> list[ItemRead]:
        items = []
        async for doc in self._collection.find():
            items.append(_doc_to_item(doc))
        return items

    async def get_one(self, item_id: str) -> Optional[ItemRead]:
        try:
            oid = ObjectId(item_id)
        except Exception:
            return None
        doc = await self._collection.find_one({"_id": oid})
        if doc is None:
            return None
        return _doc_to_item(doc)

    async def create(self, item: ItemCreate) -> ItemRead:
        result = await self._collection.insert_one(
            {"name": item.name, "description": item.description}
        )
        doc = await self._collection.find_one({"_id": result.inserted_id})
        return _doc_to_item(doc)

    async def update(self, item_id: str, item: ItemCreate) -> Optional[ItemRead]:
        try:
            oid = ObjectId(item_id)
        except Exception:
            return None
        updated = await self._collection.find_one_and_update(
            {"_id": oid},
            {"$set": {"name": item.name, "description": item.description}},
            return_document=True,
        )
        if updated is None:
            return None
        return _doc_to_item(updated)

    async def delete(self, item_id: str) -> bool:
        try:
            oid = ObjectId(item_id)
        except Exception:
            return False
        result = await self._collection.delete_one({"_id": oid})
        return result.deleted_count == 1
