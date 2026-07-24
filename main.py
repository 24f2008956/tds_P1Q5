import os
import json
import logging
import uuid
import requests
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

# 1. Load Environment Variables
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.aipipe.org/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
GIST_TOKEN = os.getenv("GITHUB_GIST_TOKEN")

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. In-memory chat history per user (for multi-turn support)
chat_histories = {}

# 3. Tools for the Agent
@tool
def fetch_url_content(url: str) -> str:
    """Fetches the text content from a public URL (e.g., CSV, JSON, or text dataset)."""
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.text[:8000] # Return first 8000 chars to avoid overwhelming the LLM
    except Exception as e:
        return f"Error fetching URL: {str(e)}"

@tool
def python_data_analysis(code: str) -> str:
    """Executes Python code for data analysis. Use pandas. The code MUST print the final answer."""
    try:
        safe_globals = {"__builtins__": __builtins__, "pd": __import__("pandas"), "io": __import__("io")}
        import io, sys
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        exec(code, safe_globals)
        
        sys.stdout = old_stdout
        return captured_output.getvalue().strip()
    except Exception as e:
        return f"Execution Error: {str(e)}"

tools = [fetch_url_content, python_data_analysis]

# 4. Log Uploader (GitHub Gist)
def upload_log_to_gist(run_id: str, log_entries: list) -> str:
    """Uploads a list of dict logs as JSONL to a public GitHub Gist."""
    jsonl_content = "\n".join(json.dumps(entry) for entry in log_entries)
    
    url = "https://api.github.com/gists"
    headers = {
        "Authorization": f"token {GIST_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    payload = {
        "description": f"TDS Bot Run Log: {run_id}",
        "public": True,
        "files": {
            f"run_{run_id}.jsonl": {
                "content": jsonl_content
            }
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        gist_data = response.json()
        filename = list(gist_data["files"].keys())[0]
        raw_url = gist_data["files"][filename]["raw_url"]
        return raw_url
    except Exception as e:
        logger.error(f"Failed to upload gist: {e}")
        return "https://example.com/log-failed.jsonl"

# 5. Telegram Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Data Analyst Bot is ready. Send me a data analysis question!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.message.from_user.id)
    user_text = update.message.text
    run_id = str(uuid.uuid4())
    
    run_log = [{"step": "start", "run_id": run_id, "user_input": user_text}]

    try:
        llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            temperature=0.0
        )

        # Create the agent using LangGraph
        system_prompt = """You are an expert Data Analyst. 
        Analyze the data provided (either inline or via URL) and answer the user's question.
        You have access to `fetch_url_content` and `python_data_analysis` tools.
        
        CRITICAL: Your final output MUST be a valid JSON object with exactly two keys: "answer" and "log_url".
        The "answer" value must match the exact shape requested in the user's prompt.
        Do NOT output markdown formatting like ```json. Output ONLY the raw JSON string.
        Example: {"answer": {"state": "Assam"}, "log_url": "https://..."}
        """
        
        agent = create_react_agent(llm, tools, prompt=system_prompt)

        if user_id not in chat_histories:
            chat_histories[user_id] = []
        
        # Build messages list
        messages = chat_histories[user_id][-4:] + [HumanMessage(content=user_text)]
        
        logger.info(f"Running agent for user {user_id}")
        run_log.append({"step": "agent_start", "model": LLM_MODEL})
        
        # Invoke the agent
        result = await agent.ainvoke({"messages": messages})
        
        # Extract the final AI message
        final_message = result["messages"][-1]
        raw_output = final_message.content if hasattr(final_message, 'content') else str(final_message)
        
        run_log.append({"step": "agent_result", "output": raw_output})

        # Update chat history
        chat_histories[user_id].append(HumanMessage(content=user_text))
        chat_histories[user_id].append(AIMessage(content=raw_output))

        # Clean and parse the output
        cleaned_output = raw_output.replace("```json", "").replace("```", "").strip()
        
        try:
            final_json = json.loads(cleaned_output)
            if "answer" not in final_json:
                final_json = {"answer": final_json}
        except json.JSONDecodeError:
            final_json = {"answer": {"text": cleaned_output}}

        # Upload log and add URL to response
        log_url = upload_log_to_gist(run_id, run_log)
        final_json["log_url"] = log_url

        response_str = json.dumps(final_json)
        await update.message.reply_text(response_str)

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        run_log.append({"step": "error", "message": str(e)})
        log_url = upload_log_to_gist(run_id, run_log)
        error_response = json.dumps({
            "answer": {"error": "Failed to process the request."},
            "log_url": log_url
        })
        await update.message.reply_text(error_response)

def main():
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not found in .env")
        return

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot is starting and polling...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()