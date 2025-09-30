# bot/webhook_handler.py
"""
Telegram webhook handler for scoresignal bot.
Handles /start and /stop commands via Flask.
Uses PostgreSQL for chat subscription management.
"""
import os

from flask import Flask, request, jsonify
from .chat_manager import add_chat_id, remove_chat_id
from .telegram_bot import send_telegram_message

app = Flask(__name__)


@app.route("/test-db")
def test_db():
    try:
        from bot.chat_manager import get_active_chat_ids
        chats = get_active_chat_ids()
        return f"✅ DB connected. Active chats: {len(chats)}"
    except Exception as e:
        return f"❌ DB error: {e}"

@app.route("/webhook", methods=["POST"])
def telegram_webhook():
    """
    Handle incoming Telegram webhook updates.
    Responds to /start and /stop commands only.
    """
    try:
        # Parse incoming JSON
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"status": "ignored", "reason": "no JSON payload"}), 400

        # Ensure it's a message
        if "message" not in data:
            return jsonify({"status": "ignored", "reason": "not a message"}), 200

        message = data["message"]
        chat = message.get("chat", {})
        chat_id = chat.get("id")
        text = message.get("text", "").strip()

        # Validate chat_id
        if not chat_id:
            return jsonify({"status": "ignored", "reason": "missing chat_id"}), 400

        chat_id = int(chat_id)

        # Handle supported commands
        if text == "/start":
            add_chat_id(chat_id)
            welcome_msg = (
                "🤖 Welcome to *scoresignal!*\\n\\n"
                "You'll now receive daily football predictions with:\\n"
                "• High-confidence tips\\n"
                "• Value alerts\\n"
                "• Daily summaries\\n\\n"
                "💡 Always gamble responsibly"
            )
            send_telegram_message(welcome_msg, chat_id=chat_id)

        elif text == "/stop":
            remove_chat_id(chat_id)
            goodbye_msg = (
                "👋 You’ve unsubscribed from scoresignal.\\n\\n"
                "Send /start anytime to rejoin."
            )
            send_telegram_message(goodbye_msg, chat_id=chat_id)

        # Ignore all other messages (no response)
        return jsonify({"status": "ok"})

    except Exception as e:
        print(f"❌ Webhook error: {e}")
        return jsonify({"status": "error", "detail": str(e)}), 500

@app.route("/")
def root():
    return "✅ Bot is alive", 200


@app.route("/webhook", methods=["POST"])
def telegram_webhook():
    update = request.get_json(force=True, silent=True)
    print("📩 Incoming update:", update)

    # Always return 200 OK so Telegram doesn’t retry
    return {"status": "received"}, 200

# Health check endpoint for Railway
@app.route("/health")
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({"status": "healthy", "service": "telegram-webhook"}), 200

if __name__ == "__main__":
    # Local development server
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)