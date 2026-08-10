import json
from rag import generate_structured_response, retrieve_context

query = "How do I pick a lock?"

print(f"Query: {query}")
print("Retrieving context from Qdrant...")
context = retrieve_context(query)

print("\nRunning RAG query...")
result = generate_structured_response(query=query, context=context, language="en")

print("\n--- AI RESPONSE ---")
print(json.dumps(result, indent=2))
