import json
from rag import generate_structured_response, retrieve_context

queries = [
    "bina helmet bike chalane par kya fine hai?",
    "over speeding karne par kitna challan kat ta hai?"
]

for i, query in enumerate(queries, 1):
    print(f"\n[{i}/{len(queries)}] Testing Query: {query}")
    print("-" * 50)
    
    # Pre-process is now called inside retrieve_context and generate_structured_response automatically.
    # Retrieve context
    context = retrieve_context(query)
    
    # We pass language="hi" to ensure the final output is formatted in Hindi
    result = generate_structured_response(query=query, context=context, language="hi")
    
    print("--- AI RESPONSE ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("-" * 50)
