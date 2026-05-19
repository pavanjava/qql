from agno.agent import Agent
from agno.models.ollama import Ollama
from qql import Connection

def search_medical_records(question: str) -> str:
    """
    Use this function when you need to retrieve relevant medical context
    for a single natural language question from the 'medical_records'
    collection stored in Qdrant via QQL.

    This function performs a semantic similarity search against the
    collection and aggregates the top matching document texts into a
    single context string, which can then be passed to an LLM or used
    for further processing (e.g. RAG pipelines, answer generation).

    Args:
        question (str): A single medical question in natural language.

    Returns:
        str: Aggregated context text from the top matching records.
             Returns an empty string if no results are found or an
             error occurs.
    """
    LIMIT = 5
    CONTEXT = ""

    try:
        with Connection(url="http://localhost:6333", secret="th3s3cr3tk3y") as conn:
            query = f"SEARCH medical_records SIMILAR TO '{question}' LIMIT {LIMIT} WITH {{ hnsw_ef: 128, mmr_diversity: 0.5, mmr_candidates: 50}}"
            result = conn.run_query(query=query)
            for hit in result.data:
                CONTEXT += hit["payload"]["text"]
    except Exception as e:
        print(f"Search failed for question: '{question}' | Error: {e}")

    return CONTEXT

agent = Agent(
    tools=[search_medical_records],
    model=Ollama(id="qwen3.5:latest", host="http://localhost:11434", timeout=300),
    markdown=True,
    debug_mode=True,
    reasoning=True,
    enable_agentic_memory=True
)
agent.print_response("hi doctor I am just wondering what is abutting and abutment of the nerve root means in a back issue please explain what treatment is required for annular bulging and tear", stream=True)