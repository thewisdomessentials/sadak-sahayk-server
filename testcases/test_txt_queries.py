import pandas as pd
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import generate_structured_response, retrieve_context, get_openai_client, CHAT_MODEL

def sanitize_test_query(query: str) -> str:
    """Uses LLM to remove the specific fine amounts and imprisonment durations from the query."""
    prompt = (
        "You are an expert test case sanitizer. The user has provided a test case that contains the EXPECTED answer within the question. "
        "Your job is to remove the specific fine amounts (e.g., '5000', '₹1000', '2000-5000') and imprisonment durations (e.g., '3 months', '1-year') from the query. "
        "Replace them with 'how much fine' or 'what duration'. "
        "CRITICAL INSTRUCTION: You MUST explicitly append 'for a 4 wheeler vehicle' to the end of the query so the backend knows the exact vehicle context. "
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
    print("Reading Testcases.txt...")
    df = pd.read_csv('Testcases.txt', sep='\t', encoding='utf-8')
    results = []
    count = 1
    
    for index, row in df.iterrows():
        original_test_case = str(row.get('Test case', ''))
        
        # Skip empty rows
        if not original_test_case or pd.isna(original_test_case) or str(original_test_case).strip() == '':
            continue
            
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
            
            # Save continuously to prevent data loss on network timeout
            with open('testcases/testcases_txt_results_v2.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error on test {count}: {e}")
        count += 1
            
    print(f"\nSuccessfully ran tests! Saved to testcases_txt_results.json")

if __name__ == "__main__":
    run_tests()
