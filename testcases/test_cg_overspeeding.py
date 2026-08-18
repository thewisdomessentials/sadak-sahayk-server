import json
import sys
import os

# Add parent directory to path so we can import rag
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import generate_structured_response, retrieve_context

query = "dVerify drunken driving challan should be  for the first offense and  for the subsequent offense as per the latest Chhattisgarh schedule as per the latest Chhattisgarh schedule"

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
