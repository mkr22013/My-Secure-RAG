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
    for item in master_map:
        # Match Year (Number or String)
        q_year = str(item["year"]).strip() == str(year).strip()
        # Match Type (Is 'Med' in 'Medical'?)
        q_type = plan_type.lower()[:3] in item["type"].lower()
        # Match Tier (Is 'Gold' in 'Gold Plan'?)
        q_tier = plan_tier.lower()[:3] in item["tier"].lower()
        
        if q_year and q_type and q_tier:
            target = item
            break
            
    if not target:
        # This will show in your terminal so you can see why it failed
        return f"ERROR: Could not find a match for Year:{q_year}, Type:{q_type}, Tier:{q_tier}"

    with open(target["sub_index_file"], "r") as f:
        sub_index = json.load(f)
    
    # Use a list of matches instead of 'next' to find the most relevant section
    best_page_data = next((p for p in sub_index if topic.lower() in p["keywords"]), None)
    
    if not best_page_data: 
        return f"INFO: Topic '{topic}' not found in {year} {plan_tier} {plan_type} booklet."

    # --- START OF SLIDING WINDOW EXTRACTION ---
    try:
        reader = PdfReader(best_page_data["file_path"])
        current_page = best_page_data["page_number"]
        total_pages = len(reader.pages)

        # We grab the page before and the page after for full context
        # max(0, ...) ensures we don't go below page 1
        # min(total_pages - 1, ...) ensures we don't go past the last page
        start_range = max(0, current_page - 1)
        end_range = min(total_pages - 1, current_page + 1)

        full_context = ""
        for p in range(start_range, end_range + 1):
            page_text = reader.pages[p].extract_text()
            full_context += f"\n--- DOCUMENT PAGE {p + 1} ---\n{page_text}\n"

        # Return the multi-page context to the LLM
        return (
            f"SOURCE: {best_page_data['file_path']}\n"
            f"MATCH FOUND ON PAGE: {current_page + 1}\n"
            f"RETRIEVED WINDOW: Pages {start_range + 1} to {end_range + 1}\n"
            f"CONTENT:\n{full_context}"
        )
    except Exception as e:
        return f"ERROR during PDF reading: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")