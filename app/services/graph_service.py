from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from datetime import datetime
import os
from collections import defaultdict
import json


class GraphServiceException(Exception):
    """Custom exception for graph service errors"""
    pass


class CustomGraphService:
    def __init__(self, api_key: str = None):
        """Initialize the service with OpenAI API key"""
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise GraphServiceException("OpenAI API key is required")
        self.llm = ChatOpenAI(
            openai_api_key=self.api_key,
            model_name="gpt-3.5-turbo",
            temperature=0
        )

    async def _extract_entities_and_relations(self, text: str) -> Dict[str, Any]:
        """Extract entities and their relationships from text using OpenAI."""
        try:
            prompt = f"""
            Analyze the following text and extract:
            1. Key entities (people, organizations, concepts, etc.)
            2. Relationships between these entities

            Text: {text}

            Return the results in JSON format:
            {{
                "entities": [
                    {{"id": "1", "name": "entity_name", "type": "entity_type"}},
                    ...
                ],
                "relationships": [
                    {{"source": "1", "target": "2", "type": "relationship_type"}},
                    ...
                ]
            }}
            """

            messages = [
                {"role": "system",
                 "content": "You are a precise entity and relationship extractor. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ]

            response = await self.llm.ainvoke(messages)
            return json.loads(response.content)

        except Exception as e:
            raise GraphServiceException(f"Failed to extract entities and relations: {str(e)}")

    async def create_knowledge_graph(self, document_id: str, chunks: List[str]) -> Dict[str, Any]:
        """
        Create a knowledge graph from document chunks.

        Args:
            document_id (str): Unique identifier for the document
            chunks (List[str]): List of text chunks from the document

        Returns:
            Dict[str, Any]: Knowledge graph data structure
        """
        try:
            all_entities = {}
            all_relationships = []

            # Process each chunk
            for i, chunk in enumerate(chunks):
                result = await self._extract_entities_and_relations(chunk)

                for entity in result['entities']:
                    new_id = f"{i}_{entity['id']}"
                    all_entities[new_id] = {
                        'name': entity['name'],
                        'type': entity['type'],
                        'chunk_index': i
                    }

                for rel in result['relationships']:
                    new_rel = {
                        'source': f"{i}_{rel['source']}",
                        'target': f"{i}_{rel['target']}",
                        'type': rel['type']
                    }
                    all_relationships.append(new_rel)

            # Create the graph structure
            graph = {
                'nodes': [{'id': k, **v} for k, v in all_entities.items()],
                'edges': all_relationships,
                'metadata': {
                    'document_id': document_id,
                    'num_chunks': len(chunks),
                    'created_at': datetime.now().isoformat()
                }
            }

            return graph

        except Exception as e:
            raise GraphServiceException(f"Failed to create knowledge graph: {str(e)}")

    async def query_knowledge_graph(
            self,
            graph: Dict[str, Any],
            query: str
    ) -> List[Dict[str, Any]]:
        """
        Query the knowledge graph.
        """
        try:
            prompt = f"""
            Given this knowledge graph:
            Nodes: {json.dumps(graph['nodes'][:10])}
            Edges: {json.dumps(graph['edges'][:10])}

            Answer this query: {query}

            Return the response in JSON format:
            {{
                "results": [
                    {{
                        "relevant_nodes": ["node_ids"],
                        "explanation": "explanation of relevance",
                        "confidence": 0.95
                    }},
                    ...
                ]
            }}
            """

            messages = [
                {"role": "system", "content": "You are a knowledge graph query analyzer. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ]

            response = await self.llm.ainvoke(messages)
            results = json.loads(response.content)

            # Enhance results with actual node data
            enhanced_results = []
            nodes_dict = {node['id']: node for node in graph['nodes']}

            for result in results['results']:
                relevant_nodes = [nodes_dict[node_id] for node_id in result['relevant_nodes'] if node_id in nodes_dict]
                enhanced_results.append({
                    'nodes': relevant_nodes,
                    'explanation': result['explanation'],
                    'confidence': result['confidence']
                })

            return enhanced_results

        except Exception as e:
            raise GraphServiceException(f"Failed to query knowledge graph: {str(e)}")

    def get_graph_statistics(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """Get basic statistics about the knowledge graph."""
        try:
            node_types = defaultdict(int)
            edge_types = defaultdict(int)

            for node in graph['nodes']:
                node_types[node['type']] += 1

            for edge in graph['edges']:
                edge_types[edge['type']] += 1

            return {
                'total_nodes': len(graph['nodes']),
                'total_edges': len(graph['edges']),
                'node_type_distribution': dict(node_types),
                'edge_type_distribution': dict(edge_types)
            }

        except Exception as e:
            raise GraphServiceException(f"Failed to get graph statistics: {str(e)}")
