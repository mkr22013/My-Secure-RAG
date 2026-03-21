import os
import json
import ollama
from pypdf import PdfReader
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
DOC_BASE_DIR = os.getenv("DOC_BASE_DIR", "./docs")
INDEX_OUTPUT_DIR = os.getenv("INDEX_OUTPUT_DIR", "./indices")
MASTER_INDEX_FILE = os.getenv("MASTER_INDEX_FILE", "master_metadata.json")
LOCAL_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

os.makedirs(INDEX_OUTPUT_DIR, exist_ok=True)

# --- STEP 1: REAL-WORLD CLASSIFICATION ---
def classify_document(pdf_path):
    """
    Identifies the document by reading the actual content of the first page.
    This ignores 'messy' filenames and finds the real Year, Type, and Tier.
    """
    try:
        reader = PdfReader(pdf_path)
        # We usually only need the first 1500 chars of page 1 for the title/year
        cover_text = reader.pages[0].extract_text()
        
        prompt = f"""
        Analyze the following text from an insurance booklet cover. 
        Extract the 'year', 'type' (Medical, Dental, or Vision), and 'tier' (Gold, Silver, or Bronze).
        If the tier is not found, use 'Standard'.
        Return ONLY a clean JSON object.
        
        TEXT:
        {cover_text[:1500]}
        """
        
        response = ollama.generate(model=LOCAL_MODEL, prompt=prompt, format="json")
        data = json.loads(response['response'])
        
        return {
            "year": int(data.get('year', 0)),
            "type": data.get('type', 'Unknown'),
            "tier": data.get('tier', 'Standard')
        }
    except Exception as e:
        print(f"Error classifying {pdf_path}: {e}")
        return None

# --- STEP 2: PAGE-BY-PAGE INDEXING ---
def generate_sub_index(pdf_path, plan_info):
    """
    Creates a detailed 'Surgical Map' for one specific booklet.
    Each page gets a summary and keywords for vectorless search.
    """
    sub_index = []
    reader = PdfReader(pdf_path)
    print(f"[*] Creating sub-index for: {plan_info['year']} {plan_info['type']} {plan_info['tier']}")

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text or len(text) < 100: continue

        # Local LLM summarizes this specific page
        prompt = f"Summarize this insurance page and list 5 keywords. Return ONLY JSON: {{'topic': '...', 'keywords': []}}. Text: {text[:1000]}"
        try:
            response = ollama.generate(model=LOCAL_MODEL, prompt=prompt, format="json")
            metadata = json.loads(response['response'])
            
            sub_index.append({
                "file_path": os.path.abspath(pdf_path),
                "page_number": page_num,
                "topic": metadata.get('topic', 'General'),
                "keywords": [k.lower() for k in metadata.get('keywords', [])]
            })
        except:
            continue

    # Filename for the sub-index (e.g., 2025_medical_gold.json)
    idx_filename = f"{plan_info['year']}_{plan_info['type']}_{plan_info['tier']}.json".lower()
    sub_index_path = os.path.join(INDEX_OUTPUT_DIR, idx_filename)
    
    with open(sub_index_path, "w") as f:
        json.dump(sub_index, f, indent=4)
        
    return sub_index_path

# --- STEP 3: MASTER ORCHESTRATION ---
def build_all():
    master_metadata = []
    
    # Iterate through all files in the docs directory (and subfolders)
    for root, dirs, files in os.walk(DOC_BASE_DIR):
        for filename in files:
            if not filename.lower().endswith(".pdf"): continue
            
            pdf_full_path = os.path.join(root, filename)
            
            # Identify the document by its actual text, not its name
            plan_info = classify_document(pdf_full_path)
            
            if plan_info and plan_info['year'] != 0:
                # Generate the detailed sub-index for this plan
                sub_path = generate_sub_index(pdf_full_path, plan_info)
                
                # Add this plan to the Global Master Index
                master_metadata.append({
                    **plan_info,
                    "sub_index_file": os.path.abspath(sub_path),
                    "original_filename": filename
                })
                print(f"Successfully indexed: {filename} as {plan_info['year']} {plan_info['type']}")

    # Save the Final Master Index
    with open(MASTER_INDEX_FILE, "w") as f:
        json.dump(master_metadata, f, indent=4)
    print(f"\n--- SUCCESS: Master Index saved to {MASTER_INDEX_FILE} ---")

if __name__ == "__main__":
    build_all()
