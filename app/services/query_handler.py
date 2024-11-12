from typing import List, Dict
from app.utils.config import db
from app.utils.embedding_utils import get_embedding


def retrieve_similar_texts(document_id: str, query: str, top_n: int = 5) -> List[Dict]:
    try:
        query_vector = get_embedding(query)
        table = db.open_table('embeddings')
        results_df = table.search(
            query_vector,
            vector_column_name="vector",
        ).where(
            f"document_id = '{document_id}'",
            prefilter=True
        ).select(
            ["text", "vector", "knowledge_graph"]
        ).limit(top_n).to_pandas()

        if results_df.empty:
            return []

        return results_df.to_dict('records')

    except Exception as e:
        raise Exception(f"Error retrieving similar texts: {str(e)}")