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
git clone https://github.com/yourusername/embedding-retrieval-system.git
cd embedding-retrieval-system
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
DATABASE_URL=your_database_url  # Example for LanceDB connection``` 
```
### 4. API Usage
This project includes endpoints to create embeddings, retrieve similar texts, and generate answers.

Create Embeddings
POST /create-embeddings
Request Body:

```    json
{
  "document_id": "doc1",
  "chunks": ["This is the first chunk.", "This is the second chunk."],
  "knowledge_graph": {"example_key": "example_value"}
}
```
Response:

```json
{
  "status": "success",
  "message": "Embeddings created successfully."
}
```
Retrieve Similar Texts
POST /retrieve-similar-texts
Request Body:

```json
{
  "document_id": "doc1",
  "query": "Sample query text",
  "top_n": 5
}
```
        
Response:

```json
[
  {
    "text": "This is the first chunk.",
    "score": 0.85,
    "knowledge_graph": "example_value"
  },
  ...
]
```
Generate Answer
POST /generate-answer
Request Body:

```json
{
  "document_id": "doc1",
  "knowledge_graph": {"example_key": "example_value"},
  "query": "What is the content about?"
}
```
Response:

```json

{
  "answer": "The document discusses..."
}
```