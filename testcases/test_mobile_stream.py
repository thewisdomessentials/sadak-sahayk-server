import os
from rag import retrieve_context, stream_response
from dotenv import load_dotenv

def test_mobile_streaming(query: str):
    load_dotenv()
    print(f"--- Testing Mobile App Streaming Flow ---")
    print(f"User Query: {query}")
    
    # 1. Retrieve context (exactly like main.py limit=10)
    print("\n[1] Retrieving Context...")
    context = retrieve_context(query, limit=10)
    
    # 2. Stream Response (exactly like main.py req.stream=True)
    print("\n[2] Streaming Response from LLM:\n")
    
    stream_generator = stream_response(
        query=query,
        context=context,
        language="en",
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        conversation_history=None
    )
    
    for chunk in stream_generator:
        # Safely print to Windows console avoiding UnicodeEncodeError (like the ₹ symbol)
        print(chunk.encode('ascii', 'ignore').decode('ascii'), end="", flush=True)
    
    print("\n\n--- Test Complete ---")

if __name__ == "__main__":
    test_cases = [
        "What is the penalty for drunk driving?"
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*50}\nTest Case {i}/{len(test_cases)}\n{'='*50}")
        test_mobile_streaming(case)
