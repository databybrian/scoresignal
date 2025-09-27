# config.py
import os
from dotenv import load_dotenv
from pathlib import Path
from datetime import date

load_dotenv()
# API Keys (use environment variables for security!)
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
# TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# API Settings
API_URL = "https://api.together.xyz/v1/chat/completions"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
HEADERS = {
    "Authorization": f"Bearer {TOGETHER_API_KEY}",
    "Content-Type": "application/json"
}

# File Paths & Dates
current_season = "2025-26"
TODAY = str(date.today())
JSON_STORAGE_FILE = "bot_messages.json"
SCRIPT_DIR = Path(__file__).parent

# Logging (standard Python logging format)
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"