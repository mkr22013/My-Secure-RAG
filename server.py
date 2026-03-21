import json, os
from fastmcp import FastMCP
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()
MASTER_INDEX = os.getenv("MASTER_INDEX_FILE")

mcp = FastMCP("Insurance-Secure-RAG")

@mcp.tool()
def query_insurance_benefits(year, plan_type, plan_tier, topic):
    with open(MASTER_INDEX, "r") as f:
        master_map = json.load(f)
    
    target = None
    # Use lowercase/strip on the LLM's incoming arguments
    q_year = str(year).strip()
    q_type = str(plan_type).lower().strip()
    q_tier = str(plan_tier).lower().strip()

    for item in master_map:
        # Match against the Index
        y_match = str(item["year"]).strip() == q_year
        # Check if 'Medical' is in 'Medical Plan'
        t_match = q_type in str(item["type"]).lower() or str(item["type"]).lower() in q_type
        tr_match = q_tier in str(item["tier"]).lower() or str(item["tier"]).lower() in q_tier
        
        if y_match and t_match and tr_match:
            target = item
            break
            
    if not target:
        # This will show in your terminal so you can see why it failed
        return f"ERROR: Could not find a match for Year:{q_year}, Type:{q_type}, Tier:{q_tier}"

    with open(target["sub_index_file"], "r") as f:
        sub_index = json.load(f)
    
    # Search the keywords we generated in indexer.py
    best_page = next((p for p in sub_index if topic.lower() in p["keywords"]), None)
    if not best_page: return f"INFO: Topic '{topic}' not found in {year} booklet."

    reader = PdfReader(best_page["file_path"])
    content = reader.pages[best_page["page_number"]].extract_text()
    return f"SOURCE: {best_page['file_path']} (Pg {best_page['page_number']})\nCONTENT: {content}"

if __name__ == "__main__": mcp.run(transport="stdio")