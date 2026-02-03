
---

## 📌 Endee RAG System

An end-to-end **Retrieval Augmented Generation (RAG) backend** built using **FastAPI** and **Endee vector database**.
The system supports document ingestion, semantic search, and context retrieval through REST APIs.

---

## 🚀 Features

* Document ingestion and vector indexing
* Semantic similarity search
* RAG-style context retrieval
* Memory optimization and monitoring
* Dockerized mock Endee service
* Fully tested backend APIs

---

## 🛠 Tech Stack

* **Backend:** FastAPI, Python
* **Vector DB:** Endee (mock service)
* **Testing:** Pytest
* **Deployment:** Docker

---

## ⚙️ Setup & Run

```bash
git clone <repo-url>
cd endee-rag-system
pip install -r requirements.txt
docker compose up
```

Start backend:

```bash
uvicorn backend.main:app --reload
```

---

## 🔗 API Endpoints

| Method | Endpoint     | Description          |
| ------ | ------------ | -------------------- |
| GET    | `/health`    | Service health check |
| POST   | `/ingest`    | Ingest documents     |
| POST   | `/search`    | Semantic search      |
| POST   | `/rag/query` | RAG-based response   |

---

## 🧪 Run Tests

```bash
python -m pytest backend/test_memory_optimization.py -v
```

---

