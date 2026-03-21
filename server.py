import json, os
from fastmcp import FastMCP
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()
MASTER_INDEX = os.getenv("MASTER_INDEX_FILE")

mcp = FastMCP("Insurance-Secure-RAG")

@mcp.tool()
def query_insurance_benefits(year: int, plan_type: str, plan_tier: str, topic: str) -> str:
    """Search specific insurance booklets by year and plan."""
    if not os.path.exists(MASTER_INDEX):
        return "ERROR: Master index not found. Run indexer.py first."

    with open(MASTER_INDEX, "r") as f:
        master_map = json.load(f)
    
    target = next((item for item in master_map if item["year"] == year and 
                   item["type"].lower() == plan_type.lower() and 
                   item["tier"].lower() == plan_tier.lower()), None)
    
    if not target: return f"ERROR: No booklet for {year} {plan_tier} {plan_type}."

    with open(target["sub_index_file"], "r") as f:
        sub_index = json.load(f)
    
    # Search the keywords we generated in indexer.py
    best_page = next((p for p in sub_index if topic.lower() in p["keywords"]), None)
    if not best_page: return f"INFO: Topic '{topic}' not found in {year} booklet."

    reader = PdfReader(best_page["file_path"])
    content = reader.pages[best_page["page_number"]].extract_text()
    return f"SOURCE: {best_page['file_path']} (Pg {best_page['page_number']})\nCONTENT: {content}"

if __name__ == "__main__": mcp.run(transport="stdio")