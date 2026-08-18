import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import generate_structured_response, retrieve_context

# We intentionally LEAVE OUT the actual fine amounts in these queries.
# If the AI returns the correct fine amounts, it proves it is actually retrieving the laws 
# from the database and not just repeating what we typed.
queries = [
    "Verify racing challan should be how much for the first offense and how much for a subsequent offense as per the latest Chhattisgarh schedule",
    "Verify drunken driving challan should be how much for the first offense and how much for the subsequent offense as per the latest Chhattisgarh schedule",
    "challan fine for overspeeding in a car should be how much as per the applicable Chhattisgarh road challan schedule"
]

for i, query in enumerate(queries, 1):
    print(f"\n[{i}/{len(queries)}] Testing Query: {query}")
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
