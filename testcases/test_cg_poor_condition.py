import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import generate_structured_response, retrieve_context

query = "Verify driving a car in poor condition should attract how much fine as per the latest Chhattisgarh schedule"

print(f"Testing Query: {query}")
print("-" * 50)

# Retrieve context
print("Retrieving context from Qdrant...")
context = retrieve_context(query)

# Generate response
print("Generating response from LLM...")
result = generate_structured_response(query=query, context=context)

print("--- AI RESPONSE ---")
print(json.dumps(result, indent=2, ensure_ascii=False))
print("-" * 50)
