# My FastAPI Project

A simple item management REST API built with FastAPI.

## Install Dependencies
```bash
pip install -r requirements.txt
```

## Run the Server
```bash
uvicorn main:app --reload
```

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /items | Return all items |
| GET | /items/{id} | Return a single item |
| POST | /items | Create a new item |
| PUT | /items/{id} | Update an existing item |
| DELETE | /items/{id} | Delete an item |