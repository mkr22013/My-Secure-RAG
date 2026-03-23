import json, os, sqlite3
from fastmcp import FastMCP
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()
DB_PATH = "insurance_index.db"

mcp = FastMCP("Insurance-Secure-RAG")

@mcp.tool()
def query_insurance_benefits(year: int, plan_type: str, plan_tier: str, topic: str) -> str:
    """Surgically retrieves benefit details using SQLite metadata filtering."""
    
    # 1. SQL-BASED METADATA FILTERING (Fast even with millions of documents)
    try:
        # Use context manager to ensure the connection always closes safely
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Fuzzy matching with LIKE to handle variations (e.g., 'Medical' vs 'Medical Plan')
            query = """
                SELECT pdf_path, sub_index_path 
                FROM master_index 
                WHERE year = ? 
                AND plan_type LIKE ? 
                AND plan_tier LIKE ?
                LIMIT 1
            """
            cursor.execute(query, (year, f"%{plan_type}%", f"%{plan_tier}%"))
            row = cursor.fetchone()
            
        if not row:
            return f"ERROR: No booklet match found in DB for {year} {plan_tier} {plan_type}."
        
        pdf_path, sub_index_file = row

    except Exception as e:
        return f"DATABASE ERROR: {str(e)}"

    # 2. SUB-INDEX LOOKUP (Surgical Pointer for the specific PDF)
    if not os.path.exists(sub_index_file):
        return f"ERROR: Sub-index file missing at {sub_index_file}"

    with open(sub_index_file, "r") as f:
        sub_index = json.load(f)
    
    # Search the keywords generated during indexing
    best_page_data = next((p for p in sub_index if topic.lower() in p["keywords"]), None)
    
    if not best_page_data: 
        return f"INFO: Topic '{topic}' not found in the {year} {plan_tier} {plan_type} document."

    # 3. CONTEXT-AWARE SLIDING WINDOW (Grabs 3 pages for full context)
    try:
        reader = PdfReader(pdf_path)
        current_page = best_page_data["page_number"]
        total_pages = len(reader.pages)

        # Boundary-aware window (prevents out-of-range crashes)
        start_range = max(0, current_page - 1)
        end_range = min(total_pages - 1, current_page + 1)

        full_context = ""
        for p in range(start_range, end_range + 1):
            page_text = reader.pages[p].extract_text()
            full_context += f"\n--- DOCUMENT PAGE {p + 1} ---\n{page_text}\n"

        return (
            f"SOURCE: {pdf_path}\n"
            f"IDENTIFIED TOPIC: {best_page_data['topic']}\n"
            f"RETRIEVED PAGES: {start_range + 1} to {end_range + 1}\n"
            f"CONTENT:\n{full_context}"
        )
    except Exception as e:
        return f"PDF ERROR: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")