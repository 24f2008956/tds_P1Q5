import os
import json
import time
import uuid
import logging
import requests

from dotenv import load_dotenv
from openai import OpenAI
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ---------------------------------------------------------
# 1. Load environment variables
# ---------------------------------------------------------

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
AIPIPE_TOKEN = os.getenv("LLM_API_KEY", "").strip()
AIPIPE_BASE_URL = os.getenv(
    "LLM_BASE_URL",
    "https://aipipe.org/openai/v1"
)
AIPIPE_MODEL = os.getenv("LLM_MODEL", "gpt-5-mini")
GIST_TOKEN = os.getenv("GITHUB_GIST_TOKEN")

# ---------------------------------------------------------
# 2. Logging
# ---------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# 3. OpenAI-compatible AI Pipe client
# ---------------------------------------------------------

client = OpenAI(
    base_url=AIPIPE_BASE_URL,
    api_key=AIPIPE_TOKEN,
)

# ---------------------------------------------------------
# 4. Conversation history
# ---------------------------------------------------------

conversation_history = {}

# ---------------------------------------------------------
# 5. Gist logging
# ---------------------------------------------------------

def upload_log_to_gist(run_id: str, log_entries: list) -> str:
    """
    Upload one run's JSONL log to a public GitHub Gist.
    """

    if not GIST_TOKEN:
        logger.error("GITHUB_GIST_TOKEN is not configured")
        return ""

    jsonl_content = "\n".join(
        json.dumps(entry, ensure_ascii=False)
        for entry in log_entries
    )

    filename = f"run_{run_id}.jsonl"

    payload = {
        "description": f"TDS Bot Run Log: {run_id}",
        "public": True,
        "files": {
            filename: {
                "content": jsonl_content
            }
        },
    }

    headers = {
        "Authorization": f"Bearer {GIST_TOKEN}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            "https://api.github.com/gists",
            headers=headers,
            json=payload,
            timeout=20,
        )

        response.raise_for_status()

        gist_data = response.json()

        raw_url = gist_data["files"][filename]["raw_url"]

        return raw_url

    except Exception as e:
        logger.exception("Failed to upload Gist: %s", e)
        return ""


# ---------------------------------------------------------
# 6. JSON extraction
# ---------------------------------------------------------

def extract_json(reply_text: str) -> dict:
    """
    Convert the model's response into a JSON object.
    """

    cleaned = reply_text.strip()

    # Remove markdown fences if the model accidentally adds them.
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "", 1)
        cleaned = cleaned.replace("```", "", 1)
        cleaned = cleaned.strip()

    # First attempt: entire response is JSON.
    try:
        parsed = json.loads(cleaned)

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        pass

    # Second attempt: extract the outermost JSON object.
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start:end + 1]

        try:
            parsed = json.loads(candidate)

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            pass

    raise ValueError("Model did not return valid JSON")


# ---------------------------------------------------------
# 7. /start
# ---------------------------------------------------------

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    await update.message.reply_text(
        "Data Analyst Bot is ready. Send me a question!"
    )


# ---------------------------------------------------------
# 8. Main message handler
# ---------------------------------------------------------

async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not update.message or not update.message.text:
        return

    chat_id = update.effective_chat.id
    user_text = update.message.text

    run_id = str(uuid.uuid4())

    run_log = [
        {
            "type": "incoming",
            "timestamp": time.time(),
            "run_id": run_id,
            "chat_id": chat_id,
            "text": user_text,
        }
    ]

    try:

        # ---------------------------------------------
        # Conversation history
        # ---------------------------------------------

        history = conversation_history.setdefault(
            chat_id,
            []
        )

        history.append(
            {
                "role": "user",
                "content": user_text,
            }
        )

        # Keep only recent messages.
        messages = history[-6:]

        # ---------------------------------------------
        # System prompt based on evaluator's solver
        # ---------------------------------------------

        system_prompt = """
You are a careful data analyst.

The user's LAST message asks a data-analysis question and
tells you exactly what JSON shape to reply with.

Your job is to work out the REAL answer.

Use reliable information available to you, including:
- MOSPI statistics when the question refers to MOSPI
- public statistical knowledge
- general knowledge
- arithmetic or calculations when required
- information contained in the conversation

IMPORTANT RULES:

1. Do NOT simply copy the JSON template from the user's question.

2. The empty values in the user's requested JSON are placeholders.
   You must replace them with the actual answer.

3. Determine the answer before producing the final response.

4. Follow the exact JSON structure requested by the user.

5. Reply with ONLY valid JSON.

6. Do NOT include markdown.

7. Do NOT include ```json fences.

8. Do NOT include explanations outside the JSON object.

9. The final response must contain exactly these two
   top-level keys:

   "answer"
   "log_url"

10. Do not invent a value when the answer can be determined
    from the available information.

Example:

If the user asks:

Which state has the highest maternal mortality rate based on
MOSPI data? Reply with ONLY this JSON object:
{"answer":{"state":""},"log_url":""}

and the correct answer is Assam, return:

{"answer":{"state":"Assam"},"log_url":""}
""".strip()

        run_log.append(
            {
                "type": "agent_start",
                "timestamp": time.time(),
                "model": AIPIPE_MODEL,
            }
        )

        # ---------------------------------------------
        # Call AI Pipe
        # ---------------------------------------------

        response = client.chat.completions.create(
            model=AIPIPE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                }
            ] + messages,
            temperature=0,
        )

        reply_text = response.choices[0].message.content.strip()

        run_log.append(
            {
                "type": "agent_output",
                "timestamp": time.time(),
                "output": reply_text,
            }
        )

        logger.info(
            "Model response for run %s: %s",
            run_id,
            reply_text,
        )

        # ---------------------------------------------
        # Save assistant response to history
        # ---------------------------------------------

        history.append(
            {
                "role": "assistant",
                "content": reply_text,
            }
        )

        # ---------------------------------------------
        # Parse JSON
        # ---------------------------------------------

        parsed = extract_json(reply_text)

        # Ensure answer exists.
        if "answer" not in parsed:
            parsed = {
                "answer": parsed
            }

        # ---------------------------------------------
        # Upload log
        # ---------------------------------------------

        log_url = upload_log_to_gist(
            run_id,
            run_log,
        )

        # ---------------------------------------------
        # Force log_url to our actual run log
        # ---------------------------------------------

        parsed["log_url"] = log_url

        final_reply = json.dumps(
            parsed,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        run_log.append(
            {
                "type": "outgoing",
                "timestamp": time.time(),
                "text": final_reply,
            }
        )

        # Note:
        # The outgoing event is added after the first Gist upload.
        # Re-upload so the final log also contains the outgoing event.
        log_url = upload_log_to_gist(
            run_id,
            run_log,
        )

        parsed["log_url"] = log_url

        final_reply = json.dumps(
            parsed,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        await update.message.reply_text(final_reply)

        logger.info(
            "Run %s completed successfully",
            run_id,
        )

    except Exception as e:

        logger.exception(
            "Error processing run %s",
            run_id,
        )

        run_log.append(
            {
                "type": "error",
                "timestamp": time.time(),
                "message": str(e),
            }
        )

        log_url = upload_log_to_gist(
            run_id,
            run_log,
        )

        error_response = {
            "answer": {
                "error": "Failed to process request"
            },
            "log_url": log_url,
        }

        await update.message.reply_text(
            json.dumps(
                error_response,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )


# ---------------------------------------------------------
# 9. Main
# ---------------------------------------------------------

def main():

    if not TELEGRAM_BOT_TOKEN:
        logger.error(
            "TELEGRAM_BOT_TOKEN is missing from .env"
        )
        return

    if not AIPIPE_TOKEN:
        logger.error(
            "LLM_API_KEY is missing from .env"
        )
        return

    logger.info("Bot is starting...")

    app = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )

    app.add_handler(
        CommandHandler("start", start)
    )

    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message,
        )
    )

    logger.info(
        "Bot is running and polling..."
    )

    app.run_polling(
        allowed_updates=Update.ALL_TYPES
    )


if __name__ == "__main__":
    main()  