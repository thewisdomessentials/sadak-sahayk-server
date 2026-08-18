import json
import os

def create_report():
    with open('testcases/txt_eval_results.json', 'r', encoding='utf-8') as f:
        results = json.load(f)
        
    md = "# Text Queries Pass/Fail Evaluation\n\n"
    md += "This report compares the AI Output against the Ground Truth rules found in `test.txt`.\n\n"
    md += "| Status | Test Case | Reason |\n|---|---|---|\n"
    
    for r in results:
        status_icon = "🟢 PASS" if "PASS" in r['status'] else "🔴 FAIL" if "FAIL" in r['status'] else "⚠️ ERROR"
        md += f"| {status_icon} | {r['test_case']} | {r['reason']} |\n"
        
    artifact_path = r"C:\Users\nites\.gemini\antigravity-ide\brain\a5f07557-1e56-4bc9-8a7e-a42dfbdc07f1\txt_evaluation_report.md"
    with open(artifact_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print("Created report!")

if __name__ == "__main__":
    create_report()
