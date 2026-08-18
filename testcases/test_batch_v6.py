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
        
        print(f"\n[{i}/{len(queries)}] Processing {tc_id}...")
        
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
            
            # Auto-answer quick replies to avoid blocking the test script
            # We'll just take the first quick reply if available to simulate a conversation
            ai_response = generate_structured_response(query=current_query, context=custom_context, conversation_history=history_str)
            
            if ai_response.get("needs_followup", False) and ai_response.get("quick_replies"):
                print("\n  ⚠️ The AI needs more information. Auto-selecting first option...")
                replies = ai_response.get("quick_replies", [])
                selected_reply = replies[0] if replies else "Yes"
                print(f"  ✅ Auto-selected: {selected_reply}")
                
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
    output_filename = "batch_query_v6_results.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2, ensure_ascii=False)
        
    print(f"\nSuccessfully processed {len(queries)} queries and saved results to {output_filename}")

if __name__ == "__main__":
    test_queries = [
        {"id": "TC-051", "query": "मैंने एक कार को रोका। चालक के पास ड्राइविंग लाइसेंस का कोई भौतिक (physical) दस्तावेज नहीं है, लेकिन आधिकारिक डेटाबेस में जांच करने पर उसका DL वैध (valid) दिखा रहा है। क्या मुझे भौतिक दस्तावेज न होने पर चालान काटना चाहिए?"},
        {"id": "TC-052", "query": "मैंने एक ट्रक को रोका। चालक के पास वैध ड्राइविंग लाइसेंस है, लेकिन वह केवल दोपहिया वाहन (LMV/MCWG) चलाने के लिए अधिकृत है, भारी मोटर वाहन (HMV) के लिए नहीं। इस स्थिति में मुझे क्या कार्रवाई करनी चाहिए?"},
        {"id": "TC-053", "query": "एक चालक का ड्राइविंग लाइसेंस 2 महीने पहले समाप्त (expired) हो गया है, जबकि दूसरे चालक का लाइसेंस अदालत द्वारा निलंबित (suspended) कर दिया गया है। इन दोनों मामलों में क्या कानूनी अंतर है और दोनों के लिए मुझे क्या कार्रवाई करनी चाहिए?"},
        {"id": "TC-054", "query": "एक वाहन का पंजीकरण प्रमाणपत्र (RC) समाप्त हो गया है, दूसरे का आरसी रद्द (cancelled) हो गया है, और तीसरे वाहन का कभी पंजीकरण ही नहीं हुआ है। इन तीनों स्थितियों में क्या कानूनी अंतर है और मुझे क्या कार्रवाई करनी चाहिए?"},
        {"id": "TC-055", "query": "एक नई कार का अस्थायी पंजीकरण (temporary registration) अभी भी वैध है, जबकि दूसरी कार का अस्थायी पंजीकरण 1 महीने पहले समाप्त हो चुका है और उसने स्थायी पंजीकरण के लिए आवेदन नहीं किया है। दोनों मामलों में मुझे क्या कार्रवाई करनी चाहिए?"},
        {"id": "TC-056", "query": "एक वाहन के निरीक्षण के दौरान, आरसी (RC) में दर्ज चेसिस नंबर वाहन पर उकेरे गए नंबर से मेल नहीं खाता है। चालक का दावा है कि यह एक लिपिकीय त्रुटि (clerical error) है। मुझे इस मामले की जांच कैसे करनी चाहिए और क्या कार्रवाई करनी चाहिए?"},
        {"id": "TC-057", "query": "एक कमर्शियल वाहन का परमिट समाप्त हो गया है, दूसरे वाहन के पास परमिट है लेकिन वह गलत रूट पर चल रहा है (परमिट शर्तों का उल्लंघन), और तीसरे वाहन के पास कोई परमिट ही नहीं है। इन तीनों उल्लंघनों में क्या अंतर है?"},
        {"id": "TC-058", "query": "एक कार 50 किमी/घंटा की सीमा वाले क्षेत्र में 75 किमी/घंटा की गति से चल रही थी, लेकिन सीधी रेखा में थी। दूसरी कार भी 75 किमी/घंटा पर थी, लेकिन वह खतरनाक तरीके से लेन बदल रही थी (weaving)। क्या दोनों को सिर्फ ओवरस्पीडिंग माना जाएगा या दूसरी कार पर खतरनाक ड्राइविंग का भी आरोप लगेगा?"},
        {"id": "TC-059", "query": "एक वाहन चालक अचानक सड़क पर आए जानवर को बचाने के लिए गलत दिशा (wrong-side) में चला गया। दूसरा चालक ट्रैफिक जाम से बचने के लिए जानबूझकर गलत दिशा में खतरनाक तरीके से ड्राइविंग कर रहा था। इन दोनों स्थितियों में मुझे क्या कार्रवाई करनी चाहिए?"},
        {"id": "TC-060", "query": "एक चालक पर शराब पीने का शक था। पहले मामले में ब्रेथ एनालाइजर टेस्ट नेगेटिव आया, दूसरे में पॉजिटिव आया (30mg/100ml से अधिक), और तीसरे मामले में चालक ने टेस्ट देने से साफ इनकार कर दिया। तीनों मामलों में मेरी क्या कार्रवाई होनी चाहिए?"},
        {"id": "TC-061", "query": "एक चालक को शराब पीकर गाड़ी चलाने (drunken driving) के लिए रोका गया। सिस्टम से पता चलता है कि उसे 1 साल पहले 'खतरनाक ड्राइविंग' (dangerous driving) के लिए दोषी ठहराया गया था। क्या यह पिछला अपराध वर्तमान शराब के अपराध के लिए 'दोहरा अपराध' (repeat offence) माना जाएगा?"},
        {"id": "TC-062", "query": "एक चालक को शराब पीकर गाड़ी चलाते हुए पकड़ा गया। रिकॉर्ड से पता चलता है कि 4 साल पहले भी उसे इसी अपराध के लिए दोषी ठहराया गया था। क्या इसे 'द्वितीय या पश्चात्वर्ती अपराध' (second or subsequent offence) माना जाएगा और जुर्माना बढ़ाया जाएगा?"},
        {"id": "TC-063", "query": "एक ट्रक को ओवरलोडिंग के संदेह में रोका गया। पहले मामले में वेइंगब्रिज (weighbridge) पर 5 टन अतिरिक्त वजन की पुष्टि हुई। दूसरे मामले में कोई वेइंगब्रिज नहीं था, केवल देखकर लगा कि ट्रक ओवरलोडेड है। क्या मैं केवल देखकर ओवरलोडिंग का चालान कर सकता हूँ?"},
        {"id": "TC-064", "query": "एक कमर्शियल यात्री वैन में अनुमति से 3 यात्री अधिक बैठे हैं, लेकिन वाहन का कुल वजन अनुमत सकल वाहन भार (GVW) के भीतर ही है। क्या मुझे अतिरिक्त यात्रियों के लिए चालान काटना चाहिए या वजन GVW के भीतर होने के कारण उन्हें छोड़ देना चाहिए?"},
        {"id": "TC-065", "query": "एक कार का बीमा (insurance) समाप्त हो चुका है। दूसरी कार का बीमा डेटाबेस में वैध (valid) दिखा रहा है, लेकिन चालक के पास कोई भौतिक पॉलिसी या डिजिटल कॉपी नहीं है। दोनों मामलों में क्या कोई उल्लंघन हुआ है?"},
        {"id": "TC-066", "query": "एक मोटरसाइकिल का पंजीकरण और बीमा पूरी तरह से वैध है, लेकिन उसका प्रदूषण नियंत्रण प्रमाणपत्र (PUC) 2 दिन पहले समाप्त हो चुका है। इस स्थिति में मुझे क्या कार्रवाई करनी चाहिए?"},
        {"id": "TC-067", "query": "एक एम्बुलेंस सायरन और बत्ती के साथ लाल बत्ती पार कर गई (आपातकाल)। दूसरी ओर, एक निजी कार केवल हैज़र्ड लाइट्स (hazard lights) चालू करके तेजी से लाल बत्ती पार कर गई। दोनों स्थितियों में मुझे क्या कार्रवाई करनी चाहिए?"},
        {"id": "TC-068", "query": "एक वाहन को एक ही समय और स्थान पर एक ही उल्लंघन के लिए दो चालान जारी किए गए हैं (डुप्लीकेट)। दूसरे मामले में, वाहन ने सुबह रेड लाइट पार की और दोपहर में ओवरस्पीडिंग की। मुझे दोनों मामलों को कैसे संभालना चाहिए?"},
        {"id": "TC-069", "query": "एक स्पीड कैमरे ने कार को ओवरस्पीडिंग करते पकड़ा, लेकिन तस्वीर में चालक का चेहरा साफ नहीं है। कार का मालिक कहता है कि वह गाड़ी नहीं चला रहा था और उसे नहीं पता कि कौन चला रहा था। इस स्थिति में चालान के लिए कौन जिम्मेदार होगा?"},
        {"id": "TC-070", "query": "एक सड़क दुर्घटना में केवल दो कारों को नुकसान (property damage) पहुँचा। दूसरी दुर्घटना में एक व्यक्ति घायल (injury) हो गया। तीसरी दुर्घटना में एक व्यक्ति की मृत्यु (death) हो गई। यातायात अधिकारी के रूप में मेरी कानूनी जिम्मेदारियां और कार्रवाई इन तीनों मामलों में कैसे भिन्न होगी?"},
        {"id": "TC-071", "query": "लापरवाही से ड्राइविंग के कारण एक व्यक्ति की मृत्यु हो गई। पहले मामले में चालक ने तुरंत पुलिस को सूचित किया और वहीं रुका रहा। दूसरे मामले में चालक मौके से भाग गया (हिट एंड रन) और पुलिस को कोई सूचना नहीं दी। दोनों मामलों में क्या अलग-अलग कानूनी कार्रवाई होगी?"},
        {"id": "TC-072", "query": "एक चालक ने जानबूझकर तेज गति से वाहन चलाकर किसी व्यक्ति को गंभीर रूप से घायल कर दिया। इसमें मोटर वाहन अधिनियम (MVA) का उल्लंघन तो है ही, साथ ही आपराधिक कानून (BNS/BNSS) के तहत भी अपराध बनता है। मुझे MVA और BNS/BNSS दोनों के तहत कब और कैसे कार्रवाई करनी चाहिए?"}
    ]
    process_batch_queries(test_queries, limit=5)
