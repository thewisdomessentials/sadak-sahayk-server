import json
from rag import generate_structured_response, retrieve_context

query = """Verify drunken driving challan should be how much fine for the first offense and for the how much fine for  subsequent offense as per the latest Chhattisgarh schedule for four wheeler / csar , also mentiom this - {
"text":"axis of the vehicle, if any, is allowed only when…"
"source":"data/AIS (400 pdf)/104201851120PM1_Draft_DF_AIS_00…"
"language":"en"
}"""

print(f"Testing Query:\n{query}\n")
print("-" * 50)

context = retrieve_context(query)
print(f"Retrieved Context Length: {len(context)} characters")

result = generate_structured_response(query=query, context=context)

with open("test_drunken_results.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
    
print("Saved outputs to test_drunken_results.json!")
