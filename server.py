import json, os, sqlite3
from fastmcp import FastMCP
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()
DB_PATH = "insurance_index.db"

mcp = FastMCP("Insurance-Secure-RAG")

@mcp.tool()
def get_available_plans() -> str:
    """
    DISCOVERY TOOL: Returns a unique list of all Plan Types, Tiers, and Years 
    currently indexed in the 100,000+ document database. 
    Use this if the user's request is vague.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            # Professional 'Distinct' lookup to map the database schema
            cursor.execute("SELECT DISTINCT year, plan_type, plan_tier FROM master_index ORDER BY year DESC")
            rows = cursor.fetchall()
            
        if not rows:
            return "DATABASE INFO: The index is currently empty."
            
        return f"DATA SOURCE SCHEMA (Year, Type, Tier): {str(rows)}"
    except Exception as e:
        return f"DISCOVERY ERROR: {str(e)}"

@mcp.tool()
def query_insurance_benefits(year: int = None, plan_type: str = None, plan_tier: str = None, topic: str = "deductible") -> str:
    """
    RETRIEVAL TOOL: Surgically retrieves benefit details using Dynamic Metadata Filtering.
    Supports broad searches if parameters are partially missing.
    """
    # 1. DYNAMIC SQL FILTERING
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            query = "SELECT year, plan_type, plan_tier, pdf_path, sub_index_path FROM master_index WHERE 1=1"
            params = []

            if year:
                query += " AND year = ?"
                params.append(year)
            if plan_type:
                query += " AND plan_type LIKE ?"
                params.append(f"%{plan_type}%")
            if plan_tier:
                query += " AND plan_tier LIKE ?"
                params.append(f"%{plan_tier}%")

            cursor.execute(query, params)
            rows = cursor.fetchall() 
            
        if not rows:
            return "ERROR: No matching plans found. Call 'get_available_plans' to see valid options."

    except Exception as e:
        return f"DATABASE ERROR: {str(e)}"

    # 2. MULTI-DOCUMENT SURGICAL EXTRACTION
    combined_results = ""
    for r_year, r_type, r_tier, pdf_path, sub_index_file in rows:
        if not os.path.exists(sub_index_file):
            continue

        with open(sub_index_file, "r") as f:
            sub_index = json.load(f)
        
        # Fuzzy keyword matching for the specific benefit topic
        best_page_data = next((
            p for p in sub_index 
            if any(topic.lower() in k.lower() or k.lower() in topic.lower() for k in p["keywords"])
        ), None)
        
        if not best_page_data:
            combined_results += f"\n--- {r_year} {r_tier} {r_type} ---\nINFO: Topic '{topic}' not found.\n"
            continue

        try:
            reader = PdfReader(pdf_path)
            current_page = best_page_data["page_number"]
            total_pages = len(reader.pages)
            start_range = max(0, current_page - 1)
            end_range = min(total_pages - 1, current_page + 1)

            page_context = ""
            for p in range(start_range, end_range + 1):
                page_text = reader.pages[p].extract_text()
                page_context += f"\n[PAGE {p + 1}]\n{page_text}\n"

            combined_results += f"\n--- {r_year} {r_tier} {r_type} ---\n{page_context}\n"
        except Exception as e:
            combined_results += f"\n--- {r_year} {r_tier} {r_type} ---\nPDF ERROR: {str(e)}\n"

    return combined_results

if __name__ == "__main__":
    mcp.run(transport="stdio")
