# bot/webhook_handler.py
"""
Telegram webhook handler for scoresignal bot.
Handles /start and /stop commands via Flask.
"""

from flask import Flask, request, jsonify
import requests

from .chat_manager import add_chat_id, remove_chat_id, ensure_table 
from .telegram_bot import send_telegram_message

app = Flask(__name__)

ensure_table()
@app.route("/webhook", methods=["POST"])
def telegram_webhook():
    """Handle incoming Telegram updates."""
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"status": "ignored", "reason": "no JSON"}), 400

        if "message" not in data:
            return jsonify({"status": "ignored", "reason": "no message"}), 200

        message = data["message"]
        chat_id = str(message["chat"]["id"])
        text = message.get("text", "").strip()

        # Handle commands
        if text == "/start":
            add_chat_id(chat_id)
            welcome_msg = (
                "🤖 Welcome to *scoresignal!*\n\n"
                "You'll now receive daily football predictions with:\n"
                "• High-confidence tips\n"
                "• Value alerts\n"
                "• Daily summaries\n\n"
                "💡 Bet responsibly"
            )
            send_telegram_message(welcome_msg, chat_id=chat_id)

        elif text == "/stop":
            remove_chat_id(chat_id)
            goodbye_msg = (
                "👋 You’ve unsubscribed from scoresignal.\n\n"
                "Send /start anytime to rejoin."
            )
            send_telegram_message(goodbye_msg, chat_id=chat_id)

        return jsonify({"status": "ok"})

    except Exception as e:
        print(f"❌ Webhook error: {e}")
        return jsonify({"status": "error", "detail": str(e)}), 500


if __name__ == "__main__":
    # Local dev mode
    app.run(host="0.0.0.0", port=5000)
