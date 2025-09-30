# bot/chat_manager.py
"""
Chat manager for handling active Telegram chat IDs using PostgreSQL (Railway DB).
"""

import os
import psycopg2
from typing import Set

DB_URL = os.getenv("DATABASE_URL")

def _get_conn():
    """Get a new database connection with SSL required for Railway."""
    if not DB_URL:
        raise RuntimeError("❌ DATABASE_URL not configured")
    # Railway requires SSL
    return psycopg2.connect(DB_URL, sslmode="require")

def ensure_table():
    """Create the telegram_chats table if it doesn't exist."""
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS telegram_chats (
                        chat_id BIGINT PRIMARY KEY,
                        username TEXT,
                        active BOOLEAN DEFAULT TRUE,
                        joined_at TIMESTAMP DEFAULT NOW()
                    )
                    """
                )
    finally:
        conn.close()

def add_chat_id(chat_id: int, username: str = None) -> None:
    """Add a new chat ID or reactivate if already exists."""
    ensure_table()  # Ensure table exists before inserting
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO telegram_chats (chat_id, username, active)
                    VALUES (%s, %s, TRUE)
                    ON CONFLICT (chat_id) DO UPDATE
                    SET active = TRUE, username = EXCLUDED.username
                    """,
                    (chat_id, username),
                )
        print(f"✅ Added/updated chat ID {chat_id}")
    except Exception as e:
        print(f"❌ Database error in add_chat_id: {e}")
        raise
    finally:
        conn.close()

def remove_chat_id(chat_id: int) -> None:
    """Mark a chat ID as inactive (unsubscribe)."""
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE telegram_chats SET active = FALSE WHERE chat_id = %s", (chat_id,))
        print(f"🗑️ Deactivated chat ID {chat_id}")
    except Exception as e:
        print(f"❌ Database error in remove_chat_id: {e}")
        raise
    finally:
        conn.close()

def get_active_chat_ids() -> Set[int]:
    """Return all currently active chat IDs."""
    ensure_table()  # Safe to call (no-op if exists)
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT chat_id FROM telegram_chats WHERE active = TRUE")
            rows = cur.fetchall()
        return {row[0] for row in rows}
    except Exception as e:
        print(f"❌ Database error in get_active_chat_ids: {e}")
        return set()
    finally:
        conn.close()