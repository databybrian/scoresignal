# bot/webhook_handler.py
"""
Telegram webhook handler for scoresignal bot.
Handles /start and /stop commands via Flask.
Uses PostgreSQL for chat subscription management.
"""

import os
import logging
from flask import Flask, request, jsonify

from bot.chat_manager import add_chat_id, remove_chat_id, get_active_chat_ids
from bot.telegram_bot import send_telegram_message

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route("/test-db")
def test_db():
    """Check DB connectivity and list active chats count."""
    try:
        chats = get_active_chat_ids()
        return f"✅ DB connected. Active chats: {len(chats)}", 200
    except Exception as e:
        logger.exception("Database connectivity error")
        return f"❌ DB error: {e}", 500


@app.route("/webhook", methods=["POST"])
def telegram_webhook():
    """
    Handle incoming Telegram webhook updates.
    Supports /start and /stop commands only (case-insensitive).
    """
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"status": "ignored", "reason": "no JSON payload"}), 400

        message = data.get("message")
        if not message:
            return jsonify({"status": "ignored", "reason": "not a message"}), 200

        chat_id = message.get("chat", {}).get("id")
        text = (message.get("text") or "").strip()

        if not chat_id:
            return jsonify({"status": "ignored", "reason": "missing chat_id"}), 400

        chat_id = int(chat_id)
        command = text.lower()  # normalize case

        if command == "/start":
            add_chat_id(chat_id)
            send_telegram_message(
                "🤖 Welcome to *scoresignal!*\n\n"
                "You'll now receive daily football predictions with:\n"
                "• High-confidence tips\n"
                "• Value alerts\n"
                "• Daily summaries\n\n"
                "💡 Always gamble responsibly",
                chat_id=chat_id,
            )
        elif command == "/stop":
            remove_chat_id(chat_id)
            send_telegram_message(
                "👋 You’ve unsubscribed from scoresignal.\n\n"
                "Send /start anytime to rejoin.",
                chat_id=chat_id,
            )
        else:
            logger.debug("Ignored message: %s", text)

        return jsonify({"status": "ok"}), 200

    except Exception as e:
        logger.exception("Webhook handler failed")
        return jsonify({"status": "error", "detail": str(e)}), 500


@app.route("/")
def root():
    """Root endpoint to confirm service is alive."""
    return "✅ Bot is alive", 200


@app.route("/health")
def health_check():
    """Health check endpoint for Railway monitoring."""
    return jsonify({"status": "healthy", "service": "telegram-webhook"}), 200


if __name__ == "__main__":
    # Local development server
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
