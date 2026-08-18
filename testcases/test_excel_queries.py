import pandas as pd
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import generate_structured_response, retrieve_context, get_openai_client, CHAT_MODEL

def is_legal_query(text):
    text = str(text).lower()
    software_keywords = ['upload', 'image', 'offline', 'camera', 'hardware', 'network', 'freeze', 'format', 'app', 'duplicate', 'device', 'double tap', 'submission']
    for kw in software_keywords:
        if kw in text: return False
    if 'verify' in text or 'challan' in text or 'fine' in text: return True
    return False

def sanitize_test_query(query: str) -> str:
    """Uses LLM to remove the specific fine amounts and imprisonment durations from the query."""
    prompt = (
        "You are an expert test case sanitizer. The user has provided a test case that contains the EXPECTED answer within the question. "
        "Your job is to remove the specific fine amounts (e.g., '5000', '₹1000', '2000-5000') and imprisonment durations (e.g., '3 months', '1-year') from the query. "
        "Replace them with 'how much fine' or 'what duration'. "
        "Return ONLY the sanitized question. Do NOT answer the question."
    )
    client = get_openai_client()
    try:
        resp = client.responses.create(
            model=CHAT_MODEL,
            max_output_tokens=100,
            input=[
                {"role": "developer", "content": prompt},
                {"role": "user", "content": query}
            ]
        )
        return resp.output_text.strip()
    except Exception:
        return query # Fallback

def run_tests():
    print("Reading Excel sheet...")
    df = pd.read_excel('Combined_Failed_Test_Cases.xlsx')
    df_subset = df.head(43)
    results = []
    count = 1
    
    for index, row in df_subset.iterrows():
        original_test_case = str(row.get('Test case', ''))
        
        if is_legal_query(original_test_case):
            # THE FIX: Strip out the numbers before testing!
            sanitized_query = sanitize_test_query(original_test_case)
            
            safe_original = original_test_case.encode('ascii', 'ignore').decode('ascii')
            safe_sanitized = sanitized_query.encode('ascii', 'ignore').decode('ascii')
            
            print(f"\n[{count}] Original: {safe_original}")
            print(f"[{count}] Sanitized: {safe_sanitized}")
            
            try:
                context = retrieve_context(sanitized_query)
                ai_result = generate_structured_response(query=sanitized_query, context=context)
                
                results.append({
                    "Original Test Case": original_test_case,
                    "Sanitized Query (Sent to AI)": sanitized_query,
                    "AI Output": ai_result
                })
            except Exception as e:
                print(f"Error on test {count}: {e}")
            count += 1
            
    with open('testcases/excel_test_results_sanitized.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"\nSuccessfully ran {count-1} sanitized queries!")

if __name__ == "__main__":
    run_tests()
