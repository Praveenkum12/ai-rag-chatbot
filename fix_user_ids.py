"""
Migration script to fix user_id for all conversations and messages
Updates NULL user_id to 'guest' for consistency
"""
from app.database import engine
from sqlalchemy import text

def fix_user_ids():
    with engine.connect() as conn:
        try:
            # Update conversations with NULL user_id
            result1 = conn.execute(text("""
                UPDATE conversations 
                SET user_id = 'guest' 
                WHERE user_id IS NULL
            """))
            
            # Update memories with NULL user_id
            result2 = conn.execute(text("""
                UPDATE user_memories 
                SET user_id = 'guest' 
                WHERE user_id IS NULL
            """))
            
            conn.commit()
            
            print(f"✓ Updated {result1.rowcount} conversations")
            print(f"✓ Updated {result2.rowcount} memories")
            print("✓ All user_id fields fixed!")
            
        except Exception as e:
            print(f"✗ Error: {e}")

if __name__ == "__main__":
    fix_user_ids()
