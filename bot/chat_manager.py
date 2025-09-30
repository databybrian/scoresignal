# bot/chat_manager.py
import os
import psycopg2
from typing import Set, Optional

# PostgreSQL connection details (set by Railway)
DB_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    """Get PostgreSQL connection."""
    try:
        conn = psycopg2.connect(DB_URL)
        return conn
    except Exception as e:
        print(f"❌ PostgreSQL connection error: {e}")
        return None

def init_db():
    """Initialize chat table if not exists."""
    conn = get_db_connection()
    if not conn:
        return
    
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS active_chats (
            chat_id BIGINT PRIMARY KEY,
            username TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()

def add_chat_id(chat_id: int, username: Optional[str] = None):
    """Add a new chat ID to active chats."""
    conn = get_db_connection()
    if not conn:
        return
    
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO active_chats (chat_id, username) 
            VALUES (%s, %s) 
            ON CONFLICT (chat_id) DO NOTHING
        """, (chat_id, username))
        conn.commit()
        print(f"✅ Added new chat ID: {chat_id}")
    except Exception as e:
        print(f"❌ Failed to add chat ID {chat_id}: {e}")
    finally:
        cursor.close()
        conn.close()

def remove_chat_id(chat_id: int):
    """Remove a chat ID from active chats."""
    conn = get_db_connection()
    if not conn:
        return
    
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM active_chats WHERE chat_id = %s", (chat_id,))
        conn.commit()
        print(f"✅ Removed chat ID: {chat_id}")
    except Exception as e:
        print(f"❌ Failed to remove chat ID {chat_id}: {e}")
    finally:
        cursor.close()
        conn.close()

def get_active_chat_ids() -> Set[int]:
    """Get all active chat IDs."""
    conn = get_db_connection()
    if not conn:
        return set()
    
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT chat_id FROM active_chats")
        rows = cursor.fetchall()
        return set(row[0] for row in rows)
    except Exception as e:
        print(f"❌ Failed to fetch active chats: {e}")
        return set()
    finally:
        cursor.close()
        conn.close()

# Initialize database on import
init_db()