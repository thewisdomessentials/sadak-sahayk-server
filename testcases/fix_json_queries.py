import json
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rag import retrieve_context, generate_structured_response

def fix_queries():
    print("Loading testcases_txt_results_v2.json...")
    with open('testcases/testcases_txt_results_v2.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    fixes_applied = 0
    for item in data:
        if item['AI Output'].get('needs_followup'):
            query = item['Sanitized Query (Sent to AI)']
            ai_answer = item['AI Output']['answer'].lower()
            
            # Figure out what it's asking based on the AI's follow-up question
            if "first offence or a repeat offence" in ai_answer or "first offence" in ai_answer:
                query += " in first offence"
            elif "private car, light motor vehicle" in ai_answer:
                query += " for a private car"
            elif "two-wheeler' or 'four-wheeler" in ai_answer or "specify the vehicle type" in ai_answer:
                # The user explicitly asked to "say four wheeler"
                query += " for a four-wheeler"
            else:
                query += " for a four wheeler vehicle"
                
            safe_query = query.encode('ascii', 'ignore').decode('ascii')
            print(f"\nFixing query: {safe_query}")
            
            try:
                context = retrieve_context(query)
                new_output = generate_structured_response(query=query, context=context)
                
                item['Sanitized Query (Sent to AI)'] = query
                item['AI Output'] = new_output
                fixes_applied += 1
            except Exception as e:
                print(f"Error re-running query: {e}")
            
    with open('testcases/testcases_txt_results_v2.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print(f"\nSuccessfully fixed {fixes_applied} queries!")

if __name__ == "__main__":
    fix_queries()
