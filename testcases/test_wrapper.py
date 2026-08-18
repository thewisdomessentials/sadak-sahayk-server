import os
from rag import preprocess_search_query, generate_structured_response, MAX_INPUT_TOKENS
from clients import get_qdrant_client, get_openai_client
from util import count_tokens, truncate_text

def run_interactive_query():
    q_client = get_qdrant_client()
    openai_client = get_openai_client()
    
    print("\n" + "="*60)
    print("🚦 Sadak Sahayak - Interactive RAG Tester 🚦")
    print("="*60)
    
    original_query = input("\nEnter your query (or 'quit' to exit): ").strip()
    if original_query.lower() in ['quit', 'exit']:
        return False
        
    current_query = original_query
    conversation_history = []
    
    while True:
        print("\n🔍 Retrieving context from Qdrant...")
        search_query = preprocess_search_query(current_query)
        if count_tokens(search_query) > MAX_INPUT_TOKENS:
            search_query = truncate_text(search_query, MAX_INPUT_TOKENS)

        emb = openai_client.embeddings.create(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
            input=[search_query],
        ).data[0].embedding

        qdrant_results = q_client.query_points(
            collection_name=os.getenv("QDRANT_COLLECTION", "sadaksahayak-documents"),
            query=emb,
            limit=20,
            with_payload=True,
            with_vectors=False,
        ).points
        
        qdrant_results.sort(key=lambda x: 0 if x.payload.get("source", "") == "data/MVA 1988 till 2025 may.pdf" else 1)
        
        custom_context_chunks = []
        for result in qdrant_results:
            source = result.payload.get("source", "Unknown")
            chunk_text = result.payload.get("text", "")
            custom_context_chunks.append(f"Source: {source}\nText: {chunk_text}")

        custom_context = "\n\n".join(custom_context_chunks)
        
        print("🧠 Generating AI response...")
        history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation_history]) if conversation_history else None
        
        ai_response = generate_structured_response(
            query=current_query, 
            context=custom_context,
            conversation_history=history_str
        )
        
        print("\n" + "-"*60)
        print("🤖 AI Output:")
        print(ai_response.get("answer", "No answer found."))
        
        if ai_response.get("needs_followup", False) and ai_response.get("quick_replies"):
            print("\n⚠️ The AI needs more information. Please select an option:")
            replies = ai_response.get("quick_replies", [])
            for i, reply in enumerate(replies, 1):
                print(f"  {i}. {reply}")
                
            while True:
                choice = input("\nEnter the number of your choice (or type your own answer): ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(replies):
                    selected_reply = replies[int(choice)-1]
                    break
                elif choice:
                    selected_reply = choice
                    break
                
            print(f"\n✅ You selected: {selected_reply}")
            
            # Save to history so LLM knows the context of the conversation
            conversation_history.append({"role": "user", "content": current_query})
            conversation_history.append({"role": "assistant", "content": ai_response.get("answer")})
            
            # Formulate the next query by appending their choice
            current_query = f"{original_query}. The user clarified: {selected_reply}."
        else:
            print("-"*60)
            print("✅ Final Answer Reached!")
            break
            
    return True

if __name__ == "__main__":
    print("Welcome to the interactive terminal testing tool!")
    while True:
        should_continue = run_interactive_query()
        if not should_continue:
            break
        
        cont = input("\nDo you want to ask another query? (y/n): ").strip().lower()
        if cont != 'y':
            break
