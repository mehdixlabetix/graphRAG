from app.services.graph_service import CustomGraphService
from app.services.query_handler import retrieve_similar_texts
from app.utils.config import Config
from langchain_openai import ChatOpenAI
from typing import Dict, List, Any

llm = ChatOpenAI(
    openai_api_key=Config.OPENAI_API_KEY,
    model_name="gpt-3.5-turbo",
    temperature=0.7
)
graph_service = CustomGraphService(api_key=Config.OPENAI_API_KEY)


async def generate_answer(document_id: str, knowledge_graph: Dict[str, Any], query: str) -> str:
    try:
        top_results = retrieve_similar_texts(document_id=document_id, query=query, )
        relevant_texts = [item['text'] for item in top_results]
        context = "\n".join(relevant_texts)

        if knowledge_graph:
            graph_results = await graph_service.query_knowledge_graph(knowledge_graph, query)
            graph_context = format_graph_results(graph_results)
        else:
            graph_context = "No knowledge graph information available."

        prompt = create_structured_prompt(context, graph_context, query)

        response = await generate_openai_response(prompt)

        return validate_and_format_answer(response)

    except Exception as e:
        return f"An error occurred while generating the answer: {str(e)}"


def format_graph_results(graph_results: List[Dict[str, Any]]) -> str:
    """Format graph results into a structured representation"""
    if not graph_results:
        return "No relevant information found in the knowledge graph."

    formatted_results = []
    for result in graph_results:
        if 'nodes' in result:
            node_info = [f"{node['name']} ({node['type']})" for node in result['nodes']]
            formatted_results.append("Related Entities: " + ", ".join(node_info))

        if 'explanation' in result:
            formatted_results.append(f"Context: {result['explanation']}")

        if 'confidence' in result:
            formatted_results.append(f"Confidence: {result['confidence']:.2f}")

    return "\n".join(formatted_results)


def create_structured_prompt(context: str, graph_context: str, query: str) -> str:
    return f"""
You are an AI assistant helping with learning materials. Answer the following question based on the provided context and knowledge graph information.

Context from Document:
{context}

Knowledge Graph Information:
{graph_context}

Question:
{query}

Instructions:
1. Use both the document context and knowledge graph information to formulate your answer
2. If the knowledge graph provides relevant relationships or connections, incorporate them
3. Answer in a clear, motivational, and pedagogical tone
4. If you cannot find sufficient information to answer the question, please explicitly state so
5. Cite specific information from either the context or knowledge graph when possible

Please provide your answer:
"""


async def generate_openai_response(prompt: str) -> str:
    messages = [
        {"role": "system",
         "content": "You are a knowledgeable educational assistant that provides accurate, well-structured answers based on provided information."},
        {"role": "user", "content": prompt}
    ]

    response = await llm.ainvoke(messages)
    return response.content.strip()


def validate_and_format_answer(response: str) -> str:
    """Validate the response and format it appropriately"""
    insufficient_info_phrases = [
        "cannot answer",
        "don't have enough information",
        "insufficient information",
        "cannot determine",
        "not enough context"
    ]

    if any(phrase in response.lower() for phrase in insufficient_info_phrases):
        return "I apologize, but I cannot provide a complete answer based on the available information in the document and knowledge graph."

    return response