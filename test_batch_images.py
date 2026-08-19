import os
import json
from pathlib import Path
from rag import answer_vision_query

def process_batch_images(image_dir="test_images", prompt=None):
    if not os.path.exists(image_dir):
        print(f"Directory '{image_dir}' not found. Creating it now...")
        os.makedirs(image_dir)
        print("Please add some images to the directory and run again.")
        return

    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = [f for f in os.listdir(image_dir) if Path(f).suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No images found in '{image_dir}'. Please add some images.")
        return

    results_list = []
    
    for i, filename in enumerate(image_files, 1):
        file_path = os.path.join(image_dir, filename)
        print(f"\n[{i}/{len(image_files)}] Processing image: {filename}")
        
        try:
            with open(file_path, "rb") as img_file:
                image_bytes = img_file.read()
            
            # Determine content type
            ext = Path(filename).suffix.lower()
            content_type = f"image/{ext[1:]}" if ext != ".jpg" else "image/jpeg"
            
            # 1. Generate AI Response using RAG vision
            print("  -> Generating AI vision analysis and RAG response...")
            response = answer_vision_query(
                image_bytes=image_bytes,
                content_type=content_type,
                language="hi", # Can change to "en" if you want English output
                prompt=prompt
            )
            
            # 2. Extract sources from context cleanly
            raw_context = response.get("context", "")
            source_lines = [line.replace("Source: ", "") for line in raw_context.split("\n") if line.startswith("Source: ")]
            unique_sources = list(dict.fromkeys(source_lines))
            
            # 3. Append to results
            results_list.append({
                "Image_Name": filename,
                "Prompt_Used": prompt,
                "Vision_Analysis": response.get("analysis"),
                "RAG_Answer": response.get("answer"),
                "Sources_Used": unique_sources
            })
            print("  [SUCCESS] Successfully analyzed image")
            
        except Exception as e:
            print(f"  [FAILED] {e}")
            results_list.append({
                "Image_Name": filename,
                "Error": str(e)
            })
        
    # Save all to JSON
    output_filename = "batch_image_results.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2, ensure_ascii=False)
        
    print(f"\nSuccessfully processed {len(image_files)} images and saved results to {output_filename}")

if __name__ == "__main__":
    # You can optionally pass a custom prompt to guide the AI
    # (e.g. "Identify all traffic violations visible in this image.")
    process_batch_images(
        image_dir="test_images", 
        prompt="Identify all traffic violations visible in this image."
    )
