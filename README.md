# Embedding and Retrieval System

This project provides an embedding-based retrieval system that generates embeddings for document chunks, stores them in a database, and retrieves relevant results based on cosine similarity. It leverages OpenAI embeddings, a knowledge graph, and LangChain OpenAI for answer generation.

## Features

- **Embedding Creation**: Generates embeddings for text chunks using OpenAI's embedding model.
- **Storage and Indexing**: Stores embeddings in LanceDB with efficient indexing for similarity search.
- **Query Retrieval**: Retrieves similar text chunks based on a query embedding.
- **Answer Generation**: Provides answers based on retrieved content and knowledge graph context.

## Requirements

- Python 3.8+
- Access to OpenAI API for embedding and language models

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/mehdixlabetix/graphRAG.git
cd graphRAG
```
### 2. Install Dependencies
Use the requirements.txt file to install the required packages.

```bash
pip install -r requirements.txt
```
### 3. Configure Environment Variables
Create a `.env` file in the root directory and add the following environment variables:

```bash
OPENAI_API_KEY=your_openai_api_key
DATABASE_URL=your_database_url  #LanceDB connection uri``` 
```

### 4. Run the Application
Start the FastAPI server using the following command:

```bash
python -m uvicorn main:app
```
   
### 5. API Usage
This project includes endpoints to create embeddings, retrieve similar texts, and generate answers.

Create Embeddings
POST /upload
Request Body:

```    json
{
    "url": "document url"
}

```
Response:

```json
{
    "document_id": "uuid",
    "message": "PDF processed, embeddings created, and knowledge graph built",
    "graph_stats":"graph statistics"
}
```

Generate Answer
POST /answer
Request Body:

```json
  {
    "document_id":"uuid",
    "query":"type your query here"

}
```
Response:

```json

{
  "answer": "The document discusses..."
}
```