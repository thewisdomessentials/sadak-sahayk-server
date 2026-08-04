import os
from dotenv import load_dotenv
load_dotenv()

from rag import retrieve_context, generate_response

query = "यदि कोई कमर्शियल वाहन बिना परमिट के चलाया जा रहा है, तो कौन-सी धारा लागू होगी?"

print("========== 1. RETRIEVING LEGAL CONTEXT FROM QDRANT ==========")
context = retrieve_context(query)
print("Retrieved Context Chunks:\n" + "-" * 60)
print(context)
print("-" * 60)

print("\n========== 2. GENERATING LEGAL RESPONSE IN HINDI (GPT-4.1-MINI) ==========")
answer = generate_response(query=query, context=context, language="hi")
print(answer)
print("=" * 60)
