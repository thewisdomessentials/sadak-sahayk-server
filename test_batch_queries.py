import os
import json
from rag import preprocess_search_query, generate_structured_response, MAX_INPUT_TOKENS
from clients import get_qdrant_client, get_openai_client
from util import count_tokens, truncate_text

def process_batch_queries(queries, limit=5):
    q_client = get_qdrant_client()
    openai_client = get_openai_client()
    
    results_list = []
    
    for i, test_case in enumerate(queries, 1):
        original_query = test_case["query"]
        tc_id = test_case["id"]
        
        print(f"\n[{i}/{len(queries)}] Processing {tc_id}: '{original_query}'")
        
        current_query = original_query
        conversation_history = []
        final_ai_response = None
        final_source_list = []
        
        while True:
            # 1. Preprocess query
            search_query = preprocess_search_query(current_query)
            if count_tokens(search_query) > MAX_INPUT_TOKENS:
                search_query = truncate_text(search_query, MAX_INPUT_TOKENS)

            # 2. Retrieve embeddings
            emb = openai_client.embeddings.create(
                model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
                input=[search_query],
            ).data[0].embedding

            # 3. Query Qdrant
            qdrant_results = q_client.query_points(
                collection_name=os.getenv("QDRANT_COLLECTION", "sadaksahayak-documents"),
                query=emb,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            ).points
            
            # Prioritize MVA
            qdrant_results.sort(key=lambda x: 0 if x.payload.get("source", "") == "data/MVA 1988 till 2025 may.pdf" else 1)
            
            # 4. Build context and extract unique sources
            custom_context_chunks = []
            source_list = []
            for result in qdrant_results:
                chunk_text = result.payload.get("text", "")
                source = result.payload.get("source", "Unknown")
                if source not in source_list:
                    source_list.append(source)
                custom_context_chunks.append(f"Source: {source}\nText: {chunk_text}")

            custom_context = "\n\n".join(custom_context_chunks)
            
            # 5. Generate AI Response
            print("  -> Generating AI response...")
            history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation_history]) if conversation_history else None
            ai_response = generate_structured_response(query=current_query, context=custom_context, conversation_history=history_str)
            
            if ai_response.get("needs_followup", False) and ai_response.get("quick_replies"):
                print("\n  ⚠️ The AI needs more information. Please select an option:")
                replies = ai_response.get("quick_replies", [])
                for idx, reply in enumerate(replies, 1):
                    print(f"    {idx}. {reply}")
                    
                while True:
                    choice = input("  Enter the number of your choice (or type your own answer): ").strip()
                    if choice.isdigit() and 1 <= int(choice) <= len(replies):
                        selected_reply = replies[int(choice)-1]
                        break
                    elif choice:
                        selected_reply = choice
                        break
                        
                print(f"  ✅ You selected: {selected_reply}")
                conversation_history.append({"role": "user", "content": current_query})
                conversation_history.append({"role": "assistant", "content": ai_response.get("answer")})
                current_query = f"{original_query}. The user clarified: {selected_reply}."
            else:
                final_ai_response = ai_response
                final_source_list = source_list
                break
            
        # 6. Append to results
        results_list.append({
            "Test_Case_ID": tc_id,
            "Original_Query": original_query,
            "AI_Output": final_ai_response,
            "Sources_Used": final_source_list
        })
        
    # Save all to JSON
    output_filename = "batch_query_results.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2, ensure_ascii=False)
        
    print(f"\nSuccessfully processed {len(queries)} queries and saved results to {output_filename}")

if __name__ == "__main__":
    # Feel free to modify these 10 queries to test any scenarios you want
    test_queries = [
        {"id": "TC-041", "query": "मैंने मोटरसाइकिल CG07AB6248 को रोका। सवार स्पष्ट रूप से चिह्नित 50 किमी/घंटा की गति सीमा वाले क्षेत्र में 82 किमी/घंटा की गति से चल रहा था, लेकिन उसने न तो टेढ़ी-मेढ़ी चलाई, न रेसिंग की, न ही टक्कर होते-होते बची और न ही कोई अन्य खतरनाक व्यवहार किया। क्या इसे अत्यधिक गति या खतरनाक ड्राइविंग माना जाना चाहिए?"},
        {"id": "TC-042", "query": "एक यातायात अधिकारी ने कार संख्या CG04MN7812 के लिए इलेक्ट्रॉनिक चालान जारी किया, लेकिन गलती से पंजीकरण संख्या के रूप में CG04MN7182 दर्ज कर दिया। उल्लंघन, स्थान, समय और कैमरे के साक्ष्य सही हैं। अधिकारी को क्या करना चाहिए? क्या दूसरा चालान बनाया जाना चाहिए?"},
        {"id": "TC-043", "query": "मैंने कमर्शियल टैक्सी CG03TX4526 को रोका। वाहन का वैध परमिट है, लेकिन परमिट में अधिकृत संचालन संबंधी विशिष्ट शर्तें हैं। निरीक्षण के दौरान, वाहन एक शर्त का उल्लंघन करता हुआ प्रतीत हुआ। चालक का कहना है कि परमिट स्वयं वैध है। मुझे किस कानूनी प्रावधान की जांच करनी चाहिए?"},
        {"id": "TC-044", "query": "एसयूवी CG05RK9184 एक दुर्घटना में शामिल थी। सीसीटीवी फुटेज से पता चलता है कि एसयूवी चालक गलत लेन में चला गया और दुर्घटना का कारण बना। पंजीकृत मालिक मौके पर मौजूद नहीं था और उसने बताया कि वाहन एक रिश्तेदार को उधार दिया गया था। पुलिस अधिकारी को चालक के रूप में किसे पहचानना चाहिए, और क्या मालिक को स्वतः ही उसी ड्राइविंग अपराध का चालान मिल सकता है?"},
        {"id": "TC-045", "query": "ट्रक CG10PL6723 के निरीक्षण के दौरान, चालक ने एक फिटनेस प्रमाण पत्र प्रस्तुत किया जिसकी वैधता तिथि में मैनुअल रूप से परिवर्तन किया गया प्रतीत होता है। आधिकारिक डेटाबेस में वैधता तिथि भिन्न दर्ज है। चालक का दावा है कि डेटाबेस को अपडेट नहीं किया गया है। अधिकारी को क्या सत्यापित करना चाहिए, और क्या इसे केवल वैधता समाप्त हो चुके दस्तावेज़ के चालान के रूप में ही माना जाना चाहिए?"},
        {"id": "TC-046", "query": "मैंने हाल ही में खरीदी गई कार CG04TR4521 को रोका। वाहन का अस्थायी पंजीकरण है, लेकिन उसकी वैधता अवधि समाप्त हो चुकी है। मालिक का कहना है कि स्थायी पंजीकरण के लिए आवेदन जमा कर दिया गया है और उन्होंने आवेदन की रसीद भी दिखाई। मुझे कौन-सी कानूनी प्रक्रिया की जाँच करनी चाहिए?"},
        {"id": "TC-047", "query": "मैंने मोटरसाइकिल CG07KL6193 को रोका। सवार ने हेलमेट नहीं पहना था, उसके साथ दो और यात्री सवार थे, और वह मोटरसाइकिल चलाते समय मोबाइल फोन का इस्तेमाल कर रहा था। ये सभी उल्लंघन एक ही बार में देखे गए। क्या अधिकारी को एक ही चालान जारी करना चाहिए या प्रत्येक उल्लंघन को अलग-अलग दर्ज करना चाहिए?"},
        {"id": "TC-048", "query": "एक ट्रैफिक कैमरे में कार CG05AB8274 को लाल सिग्नल पार करते हुए रिकॉर्ड किया गया है। तस्वीर में रजिस्ट्रेशन नंबर स्पष्ट रूप से दिखाई दे रहा है, लेकिन ड्राइवर का चेहरा नहीं पहचाना जा सकता। पंजीकृत मालिक का कहना है कि परिवार के कई सदस्य इस कार का इस्तेमाल करते हैं। इलेक्ट्रॉनिक चालान जारी करने से पहले अधिकारी को क्या सत्यापित करना चाहिए?"},
        {"id": "TC-049", "query": "मैंने ट्रक CG10PQ4527 को रोका क्योंकि यह अत्यधिक भार से लदा हुआ प्रतीत हो रहा था। चालक का कहना है कि माल अनुमत भार सीमा के भीतर है। मेरे पास वेइंगब्रिज रीडिंग या अन्य कोई विश्वसनीय माप उपलब्ध नहीं है। क्या मैं केवल दृश्य निरीक्षण के आधार पर ओवरलोडिंग का चालान जारी कर सकता हूँ?"},
        {"id": "TC-050", "query": "मैंने यातायात नियम तोड़ने के लिए ऑटो-रिक्शा CG06RT7815 को रोका। आधिकारिक सिस्टम में उसी उल्लंघन, उसी स्थान और उसी तारीख/समय के लिए चालान दर्ज है और भुगतान की स्थिति भी पहले से ही भुगतान की हुई है। चालक ने भुगतान की रसीद दिखाई। क्या मुझे एक और चालान जारी करना चाहिए?"},
    ]
    process_batch_queries(test_queries, limit=5)
