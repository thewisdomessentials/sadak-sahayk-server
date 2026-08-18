import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import get_openai_client, CHAT_MODEL

def evaluate():
    print("Loading Ground Truth test.txt...")
    with open('testcases/test.txt', 'r', encoding='utf-8') as f:
        ground_truth_context = f.read()
        
    print("Loading AI Outputs from testcases_txt_results_v2.json...")
    with open('testcases/testcases_txt_results_v2.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    client = get_openai_client()
    results = []
    
    print(f"Evaluating {len(data)} test cases using LLM Judge...")
    for i, item in enumerate(data):
        test_case = item['Original Test Case']
        ai_output = item['AI Output']['answer']
        
        prompt = f"""You are an expert QA judge. 
Your task is to determine if the AI's output correctly matches the Ground Truth rules for the specific violation mentioned in the test case.

Ground Truth Rules (from test.txt):
{ground_truth_context}

---
Test Case: {test_case}
AI Output: {ai_output}

Compare the AI Output against the relevant rule in the Ground Truth. 
Does the AI Output accurately reflect the fine amounts, imprisonment durations, and other penalties mentioned in the Ground Truth for this violation?
(Note: If the AI provides MORE accurate/modern information that contradicts the Ground Truth, or if it provides the exact same information, consider it a PASS. If the AI hallucinates completely wrong numbers, it is a FAIL).

Return exactly in this format:
RESULT: [PASS or FAIL]
REASON: [1 sentence explanation]
"""
        
        try:
            resp = client.responses.create(
                model=CHAT_MODEL,
                max_output_tokens=200,
                input=[{"role": "user", "content": prompt}]
            )
            eval_text = resp.output_text.strip()
            
            # Parse PASS/FAIL
            result_line = [line for line in eval_text.split('\n') if line.startswith('RESULT:')]
            reason_line = [line for line in eval_text.split('\n') if line.startswith('REASON:')]
            
            if result_line and reason_line:
                status = result_line[0].replace('RESULT:', '').strip()
                reason = reason_line[0].replace('REASON:', '').strip()
            else:
                status = "ERROR"
                reason = "Failed to parse LLM output."
            
            results.append({
                "test_case": test_case,
                "status": status,
                "reason": reason
            })
            safe_status = status.encode('ascii', 'ignore').decode('ascii')
            print(f"[{i+1}/{len(data)}] {safe_status}")
            
        except Exception as e:
            print(f"[{i+1}/{len(data)}] ERROR: {e}")
            results.append({
                "test_case": test_case,
                "status": "ERROR",
                "reason": str(e)
            })
            
    with open('testcases/txt_eval_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print("\nEvaluation complete! Saved to testcases/txt_eval_results.json")

if __name__ == "__main__":
    evaluate()
