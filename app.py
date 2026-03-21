import gradio as gr
import asyncio
from client import get_ai_response

def chat_interface(message, history):
    try:
        # Runs the async 'get_ai_response' function from your client.py
        return asyncio.run(get_ai_response(message))
    except Exception as e:
        return f"System Error: {str(e)}"

# Define the Interface (Removed the 'theme' argument to fix the error)
demo = gr.ChatInterface(
    fn=chat_interface, 
    title="🏢 Insurance Policy Assistant",
    description="Ask about Medical, Dental, or Vision plans (2024-2025).",
    examples=[
        "How did my Gold medical deductible change from 2024 to 2025?",
        "Does the Silver Dental plan cover braces?",
        "Compare the 2024 and 2025 family deductibles."
    ]
)

if __name__ == "__main__":
    demo.launch()