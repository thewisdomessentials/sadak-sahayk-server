import pandas as pd
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import generate_structured_response, retrieve_context

def run_tests():
    print("Reading Excel sheet (Rows 44-63)...")
    df = pd.read_excel('Combined_Failed_Test_Cases.xlsx')
    
    # Rows 44-63 in Excel correspond to index 42 to 62 in pandas (excluding header)
    # The actual queries are in the 'Remarks' column (which is the "Copy-Paste User Query" column for this section)
    # The expected section is in 'Result Status' (MVA) and 'Unnamed: 5' (CMVR)
    
    # The actual data starts from index 44 (which is row 46 in Excel, wait, let's just use the JSON file)
    with open('rows44_63.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    results = []
    
    # The first two rows of this data are the table separator and the nested headers
    # So we iterate from index 2 onwards
    for i in range(2, len(data)):
        row = data[i]
        tc_id = row.get("Test case", "")
        language = row.get("Expected", "")
        query = row.get("Remarks", "")
        expected_mva = row.get("Result Status", "")
        expected_cmvr = row.get("Unnamed: 5", "")
        
        if not query:
            continue
            
        safe_query = query.encode('ascii', 'ignore').decode('ascii')
        print(f"\n[{tc_id}] Running {language} Query: {safe_query}")
        
        try:
            # We don't need to sanitize these because these are natural user queries without leading fake amounts!
            context = retrieve_context(query)
            ai_result = generate_structured_response(query=query, context=context)
            
            results.append({
                "TC_ID": tc_id,
                "Language": language,
                "User Query": query,
                "Expected MVA": expected_mva,
                "Expected CMVR": expected_cmvr,
                "AI Output": ai_result
            })
            
            # Write continuously to avoid losing data on network timeout
            with open('testcases/multilingual_test_results.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error on test {tc_id}: {e}")
            
    with open('testcases/multilingual_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"\nSuccessfully ran {len(results)} multilingual queries!")

if __name__ == "__main__":
    run_tests()
