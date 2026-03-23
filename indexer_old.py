import os, json, ollama
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

DOC_BASE_DIR = os.getenv("DOC_BASE_DIR")
INDEX_OUTPUT_DIR = os.getenv("INDEX_OUTPUT_DIR")
MASTER_INDEX_FILE = os.getenv("MASTER_INDEX_FILE")
LOCAL_MODEL = os.getenv("OLLAMA_MODEL")

os.makedirs(INDEX_OUTPUT_DIR, exist_ok=True)

def generate_sub_index(pdf_path, plan_info):
    sub_index = []
    reader = PdfReader(pdf_path)
    print(f"[*] Indexing Page-by-Page: {pdf_path}")
    
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text or len(text) < 100: continue

        prompt = f"Identify the insurance topic and 5 keywords for this page. Return ONLY JSON: {{'topic': '...', 'keywords': []}}. Text: {text[:1200]}"
        try:
            response = ollama.generate(model=LOCAL_MODEL, prompt=prompt, format="json")
            metadata = json.loads(response['response'])
            sub_index.append({
                "file_path": os.path.abspath(pdf_path),
                "page_number": page_num,
                "topic": metadata.get('topic', 'General'),
                "keywords": [k.lower() for k in metadata.get('keywords', [])]
            })
        except: continue

    file_name = f"{plan_info['year']}_{plan_info['type']}_{plan_info['tier']}.json".lower()
    sub_index_path = os.path.join(INDEX_OUTPUT_DIR, file_name)
    with open(sub_index_path, "w") as f:
        json.dump(sub_index, f, indent=4)
    return sub_index_path

def build_all():
    master_metadata = []
    if not os.path.exists(DOC_BASE_DIR):
        print(f"Error: {DOC_BASE_DIR} folder not found. Run dummy script (create_test_docs.py) first!")
        return

    for year_folder in os.listdir(DOC_BASE_DIR):
        year_path = os.path.join(DOC_BASE_DIR, year_folder)
        if not os.path.isdir(year_path): continue
        for filename in os.listdir(year_path):
            if not filename.endswith(".pdf"): continue
            parts = filename.replace(".pdf", "").split("_")
            plan_info = {"year": int(year_folder), "type": parts[0], "tier": parts[1] if len(parts)>1 else "Gold"}
            sub_path = generate_sub_index(os.path.join(year_path, filename), plan_info)
            master_metadata.append({**plan_info, "sub_index_file": os.path.abspath(sub_path)})
    
    with open(MASTER_INDEX_FILE, "w") as f:
        json.dump(master_metadata, f, indent=4)
    print("--- Master Indexing Complete ---")

if __name__ == "__main__": build_all()




