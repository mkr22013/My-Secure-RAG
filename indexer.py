import os, json, sqlite3, ollama, re
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()
DOC_BASE_DIR = os.getenv("DOC_BASE_DIR", "./docs")
INDEX_OUTPUT_DIR = os.getenv("INDEX_OUTPUT_DIR", "./indices")
DB_PATH = "insurance_index.db"
LOCAL_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

os.makedirs(INDEX_OUTPUT_DIR, exist_ok=True)

def setup_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS master_index') # Start totally fresh
    cursor.execute('''
        CREATE TABLE master_index (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            year INTEGER, plan_type TEXT, plan_tier TEXT,
            pdf_path TEXT, sub_index_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

def nuclear_flatten(val):
    """
    Guarantees a clean string. If it's a list, it takes the first item.
    It then removes ALL brackets, quotes, and non-alphanumeric junk.
    """
    if isinstance(val, list):
        val = val[0] if len(val) > 0 else "Unknown"
    
    # Convert to string and remove ANY characters that aren't letters, numbers, or spaces
    clean = re.sub(r'[^a-zA-Z0-9\s]', '', str(val))
    return clean.strip()

def classify_document(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        cover_text = reader.pages[0].extract_text()
        
        prompt = f"Extract 'year', 'type' (Medical, Dental, Vision), and 'tier' (Gold, Silver, Bronze). Return ONLY JSON. Text: {cover_text[:1000]}"
        response = ollama.generate(model=LOCAL_MODEL, prompt=prompt, format="json")
        data = json.loads(response['response'])
        
        # --- THE NUCLEAR FIX ---
        raw_year = nuclear_flatten(data.get('year', '0'))
        # Extract ONLY the digits for the year (fixes 'Plan Year 2025' -> '2025')
        clean_year = re.sub(r'\D', '', raw_year)
        
        return {
            "year": int(clean_year) if clean_year else 0,
            "type": nuclear_flatten(data.get('type', 'Medical')),
            "tier": nuclear_flatten(data.get('tier', 'Standard'))
        }
    except Exception as e:
        print(f"Error classifying {pdf_path}: {e}")
        return None

def generate_sub_index(pdf_path, plan_info):
    sub_index = []
    reader = PdfReader(pdf_path)
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text or len(text) < 50: continue
        
        prompt = f"Topic and 5 keywords for this page. Return ONLY JSON: {{'topic': '...', 'keywords': []}}. Text: {text[:800]}"
        try:
            response = ollama.generate(model=LOCAL_MODEL, prompt=prompt, format="json")
            metadata = json.loads(response['response'])
            
            sub_index.append({
                "page_number": page_num,
                "topic": nuclear_flatten(metadata.get('topic', 'General')),
                "keywords": [nuclear_flatten(k).lower() for k in metadata.get('keywords', [])]
            })
        except: continue
    
    # Create clean filename
    y, t, tr = plan_info['year'], plan_info['type'].lower(), plan_info['tier'].lower()
    clean_fn = f"{y}_{t}_{tr}.json"
    
    sub_index_path = os.path.join(INDEX_OUTPUT_DIR, clean_fn)
    with open(sub_index_path, "w") as f:
        json.dump(sub_index, f, indent=4)
    return sub_index_path

def build_all():
    setup_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    for root, _, files in os.walk(DOC_BASE_DIR):
        for filename in files:
            if not filename.lower().endswith(".pdf"): continue
            
            pdf_path = os.path.join(root, filename)
            plan_info = classify_document(pdf_path)
            
            if plan_info and plan_info['year'] != 0:
                sub_path = generate_sub_index(pdf_path, plan_info)
                
                cursor.execute('''
                    INSERT INTO master_index (year, plan_type, plan_tier, pdf_path, sub_index_path)
                    VALUES (?, ?, ?, ?, ?)
                ''', (plan_info['year'], plan_info['type'], plan_info['tier'], os.path.abspath(pdf_path), os.path.abspath(sub_path)))
                print(f"✅ CLEAN INDEX: {plan_info['year']} {plan_info['type']} {plan_info['tier']}")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    # 1. DELETE OLD FILES MANUALLY FIRST
    print("[*] NUKING OLD DATA...")
    build_all()
    print("[*] DONE. CHECK YOUR /indices FOLDER!")
