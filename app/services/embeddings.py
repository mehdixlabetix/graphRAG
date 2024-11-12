import logging
from typing import List
from tqdm import tqdm
from app.utils.config import db
from app.utils.embedding_utils import get_embedding
import pyarrow as pa
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_embeddings(document_id: str, chunks: List[str], knowledge_graph):
    """Create and store embeddings in LanceDB"""
    if not chunks:
        logging.error("Chunks list is empty")
        raise ValueError("Chunks list is empty")

    try:
        embeddings = []
        for idx, chunk in enumerate(tqdm(chunks, desc="Creating embeddings")):
            embedding = get_embedding(chunk)
            embedding_array = np.array(embedding, dtype=np.float32)

            embeddings.append({
                'id': f'{document_id}_{idx}',
                'document_id': document_id,
                'text': chunk,
                'vector': embedding_array.tolist(),
                'knowledge_graph': knowledge_graph
            })

        vector_dimension = len(embeddings[0]['vector'])
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("document_id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), vector_dimension)),
            pa.field("knowledge_graph", pa.string())
        ])

        table_name = 'embeddings'
        table = db.create_table(
            table_name,
            schema=schema,
            mode="overwrite"
        )

        # Add embeddings
        table.add(embeddings)

        logger.info(f"Successfully created embeddings table with schema: {table.schema}")
        return True

    except Exception as e:
        logger.error(f"Error in create_embeddings: {str(e)}")
        raise