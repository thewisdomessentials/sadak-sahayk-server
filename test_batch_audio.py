import os
import json
from pathlib import Path
from clients import get_openai_client
from rag import retrieve_context, generate_structured_response

def transcribe_audio_direct(file_path):
    openai_client = get_openai_client()
    with open(file_path, "rb") as audio_file:
        transcript = openai_client.audio.transcriptions.create(
            model=os.getenv("OPENAI_TRANSCRIBE_MODEL", "whisper-1"),
            file=audio_file
        )
        return transcript.text

def process_batch_audio(audio_dir="test_audio", language="hi"):
    if not os.path.exists(audio_dir):
        print(f"Directory '{audio_dir}' not found. Creating it now...")
        os.makedirs(audio_dir)
        print("Please add some audio files (.mp3, .wav, .m4a) to the directory and run again.")
        return

    audio_extensions = {".mp3", ".wav", ".m4a", ".ogg", ".webm"}
    audio_files = [f for f in os.listdir(audio_dir) if Path(f).suffix.lower() in audio_extensions]
    
    if not audio_files:
        print(f"No audio files found in '{audio_dir}'. Please add some audio files.")
        return

    results_list = []
    
    for i, filename in enumerate(audio_files, 1):
        file_path = os.path.join(audio_dir, filename)
        print(f"\n[{i}/{len(audio_files)}] Processing audio: {filename}")
        
        try:
            # 1. Transcribe audio
            print("  -> Transcribing audio with Whisper...")
            transcription = transcribe_audio_direct(file_path)
            print(f"  [TRANSCRIPT]: {transcription}")
            
            # 2. Retrieve Context
            print("  -> Retrieving legal context from Qdrant...")
            context = retrieve_context(transcription)
            
            # Extract unique sources for logging
            source_lines = [line.replace("Source: ", "") for line in context.split("\n") if line.startswith("Source: ")]
            unique_sources = list(dict.fromkeys(source_lines))
            
            # 3. Generate AI Response
            print("  -> Generating structured RAG response...")
            ai_response = generate_structured_response(
                query=transcription,
                context=context,
                language=language
            )
            
            # 4. Append to results
            results_list.append({
                "Audio_File": filename,
                "Transcript": transcription,
                "RAG_Answer": ai_response.get("answer", ai_response),
                "Sources_Used": unique_sources
            })
            print("  [SUCCESS] Successfully analyzed audio")
            
        except Exception as e:
            print(f"  [FAILED] {e}")
            results_list.append({
                "Audio_File": filename,
                "Error": str(e)
            })
        
    # Save all to JSON
    output_filename = "batch_audio_results.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2, ensure_ascii=False)
        
    print(f"\nSuccessfully processed {len(audio_files)} audio files and saved results to {output_filename}")

if __name__ == "__main__":
    # You can change language to "en" if you want English output
    process_batch_audio(audio_dir="test_audio", language="hi")
