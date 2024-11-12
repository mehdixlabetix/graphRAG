from fastapi import APIRouter, HTTPException
from app.models.models import AnswerRequest, RetrieveRequest, UploadRequest
from app.services.answer_service import generate_answer
from app.services.pdf_processor import download_pdf, extract_text_from_pdf
from app.services.embeddings import create_embeddings
from app.services.graph_service import CustomGraphService
from app.utils.text_splitter import split_text
from app.utils.config import db, Config
import os
import json
import logging
import pandas as pd

router = APIRouter()
LOG = logging.getLogger(__name__)

graph_service = CustomGraphService(api_key=Config.OPENAI_API_KEY)


@router.post("/upload")
async def upload_pdf(request: UploadRequest):
    try:
        pdf_path, document_id = download_pdf(request.url)
        doc_text = extract_text_from_pdf(pdf_path)

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error during PDF processing: {str(e)}")

    chunks = split_text(doc_text)

    try:
        knowledge_graph = await graph_service.create_knowledge_graph(document_id, chunks)
        stats = graph_service.get_graph_statistics(knowledge_graph)
        LOG.info("Graph created successfully with stats: %s", stats)
        knowledge_graph_json = json.dumps(knowledge_graph)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create knowledge graph: {str(e)}")

    try:
        create_embeddings(document_id, chunks, knowledge_graph_json)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create embeddings: {str(e)}")

    os.remove(pdf_path)

    return {
        "document_id": document_id,
        "message": "PDF processed, embeddings created, and knowledge graph built",
        "graph_stats": stats
    }


@router.post("/answer")
async def answer_query(request: AnswerRequest):
    try:
        table = db.open_table('embeddings')
        df = table.to_pandas()
        document_rows = df[df['document_id'] == request.document_id]

        if document_rows.empty:
            LOG.error("No rows found for document_id: %s", request.document_id)
            raise HTTPException(status_code=404, detail="Document ID not found")

        try:
            first_row = document_rows.iloc[0]
            LOG.debug("First row keys: %s", first_row.index.tolist())

            if 'knowledge_graph' not in first_row:
                LOG.error("knowledge_graph column not found in DataFrame")
                raise HTTPException(
                    status_code=500,
                    detail="Database schema error: knowledge_graph column not found"
                )

            knowledge_graph_json = first_row['knowledge_graph']

            if pd.isna(knowledge_graph_json):
                LOG.warning("knowledge_graph is None/NaN for document_id: %s", request.document_id)
                knowledge_graph = {}  # Provide empty dict as fallback
            else:
                try:
                    knowledge_graph = json.loads(knowledge_graph_json)
                    LOG.debug("Successfully parsed knowledge graph JSON")
                except json.JSONDecodeError as json_err:
                    LOG.error("Failed to parse knowledge graph JSON: %s", str(json_err))
                    raise HTTPException(
                        status_code=500,
                        detail="Invalid knowledge graph JSON format"
                    )

        except IndexError as idx_err:
            LOG.error("Failed to access first row: %s", str(idx_err))
            raise HTTPException(
                status_code=500,
                detail="Failed to access document data"
            )

        LOG.debug("Generating answer with knowledge graph of size: %d bytes",
                  len(str(knowledge_graph)))
        answer = await generate_answer(request.document_id, knowledge_graph, request.query)

        LOG.info("Successfully generated answer for document %s", request.document_id)
        return {"answer": answer}

    except Exception as e:
        LOG.error("Error generating answer: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate answer: {str(e)}"
        )