import os
import json
from rag import preprocess_search_query, generate_structured_response, MAX_INPUT_TOKENS
from clients import get_qdrant_client, get_openai_client
from util import count_tokens, truncate_text

def test_query_with_sources(query: str, limit: int = 20):
    q_client = get_qdrant_client()
    openai_client = get_openai_client()
    
    search_query = preprocess_search_query(query)
    
    if count_tokens(search_query) > MAX_INPUT_TOKENS:
        search_query = truncate_text(search_query, MAX_INPUT_TOKENS)

    print(f"Retrieving top {limit} context chunks from Qdrant...")
    emb = openai_client.embeddings.create(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
        input=[search_query],
    ).data[0].embedding

    results = q_client.query_points(
        collection_name=os.getenv("QDRANT_COLLECTION", "sadaksahayak-documents"),
        query=emb,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    ).points

    custom_context_chunks = []
    source_list = []
    
    for result in results:
        chunk_text = result.payload.get("text", "")
        source = result.payload.get("source", "Unknown")
        score = getattr(result, 'score', None)
        
        # Capturing exactly the format you requested, plus the similarity score
        source_data = {
            "text": chunk_text,
            "source": source,
            "language": result.payload.get("language", "en"),
            "similarity_score": score
        }
        source_list.append(source_data)
            
        custom_context_chunks.append(f"Source: {source}\nText: {chunk_text}")

    custom_context = "\n\n".join(custom_context_chunks)
    
    print("Generating AI response...")
    ai_response = generate_structured_response(query=query, context=custom_context)
    
    final_output = {
        "AI Output": ai_response,
        "Top_20_Sources": source_list
    }
    
    with open("final_query_with_sources.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
        
    print("Saved final LLM output with top 20 sources to final_query_with_sources.json!")

if __name__ == "__main__":
    test_query = "What is the penalty for drunk driving?"
    test_query_with_sources(test_query, limit=20)
