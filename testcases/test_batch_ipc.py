import json
from rag import generate_structured_response, retrieve_context

queries = [
    "what is ipc 279",
    "penalty for ipc 304A",
    "ipc 337 meaning",
    "ipc 186 during traffic stop",
    "what is the fine for ipc 427"
]

for i, query in enumerate(queries, 1):
    print(f"\n[{i}/{len(queries)}] Testing Query: {query}")
    print("-" * 50)
    
    # We call generate_structured_response directly, which handles translation internally
    # But wait, in test_bns_query.py we were calling retrieve_context manually first.
    # In production, the API endpoint will likely just call generate_structured_response or similar.
    # But generate_structured_response takes 'context' as an argument.
    # We need to retrieve context first!
    context = retrieve_context(query)
    
    result = generate_structured_response(query=query, context=context, language="en")
    
    print("--- AI RESPONSE ---")
    print(json.dumps(result, indent=2))
    print("-" * 50)
