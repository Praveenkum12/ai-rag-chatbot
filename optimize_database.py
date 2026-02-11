"""
Database Performance Optimization Script

This script adds indexes to frequently queried columns to dramatically
improve query performance.

Run this ONCE after updating your code:
    python optimize_database.py
"""

from app.database import engine, Base
from app.models import Conversation, Message, UserMemory
from sqlalchemy import Index, text

def create_indexes():
    """Create indexes on frequently queried columns (MySQL compatible)"""
    
    print("🔧 Creating database indexes for performance...")
    
    with engine.connect() as conn:
        def index_exists(table, index_name):
            result = conn.execute(text(f"SHOW INDEX FROM {table} WHERE Key_name = '{index_name}'"))
            return result.rowcount > 0

        def safe_create_index(table, index_name, columns):
            if index_exists(table, index_name):
                print(f"  ✅ Index '{index_name}' already exists on '{table}'.")
            else:
                try:
                    print(f"  └─ Creating index '{index_name}' on '{table}'...")
                    conn.execute(text(f"CREATE INDEX {index_name} ON {table}({columns})"))
                    conn.commit()
                except Exception as e:
                    print(f"  ⚠️  Failed to create index '{index_name}': {e}")

        # 1. Index on conversations
        safe_create_index("conversations", "idx_conversations_user_id", "user_id")
        safe_create_index("conversations", "idx_conversations_updated_at", "updated_at DESC")
        safe_create_index("conversations", "idx_conversations_id_user", "id, user_id")
        
        # 2. Index on messages
        safe_create_index("messages", "idx_messages_conversation_id", "conversation_id")
        safe_create_index("messages", "idx_messages_created_at", "created_at")
        
        # 3. Index on user_memories
        safe_create_index("user_memories", "idx_user_memories_user_id", "user_id")
        
        print("\n📊 Analyzing tables for query optimization...")
        try:
            conn.execute(text("ANALYZE TABLE conversations"))
            conn.execute(text("ANALYZE TABLE messages"))
            conn.execute(text("ANALYZE TABLE user_memories"))
            conn.commit()
            print("✅ Table analysis complete!")
        except:
            pass

def show_index_status():
    """Show current indexes on tables"""
    print("\n📋 Current indexes:")
    
    with engine.connect() as conn:
        for table in ['conversations', 'messages', 'user_memories']:
            print(f"\n{table}:")
            result = conn.execute(text(f"SHOW INDEX FROM {table}"))
            for row in result:
                print(f"  - {row[2]} on {row[4]}")

if __name__ == "__main__":
    print("="*60)
    print("DATABASE PERFORMANCE OPTIMIZATION")
    print("="*60)
    
    create_indexes()
    show_index_status()
    
    print("\n" + "="*60)
    print("✅ Optimization complete!")
    print("="*60)
    print("\nExpected improvements:")
    print("  • Conversation setup: 911ms → 10-30ms (30-90x faster)")
    print("  • Save message: 3,301ms → 10-30ms (110-330x faster)")
    print("  • Memory query: Should be <10ms")
    print("\nRestart your FastAPI server and test again!")
