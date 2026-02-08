"""
Fix user_memories table - remove foreign key constraint
"""
from app.database import engine
from sqlalchemy import text

def fix_user_memories_table():
    with engine.connect() as conn:
        try:
            # Drop the existing table
            print("Dropping old user_memories table...")
            conn.execute(text("DROP TABLE IF EXISTS user_memories"))
            conn.commit()
            
            # Recreate without foreign key constraint
            print("Creating new user_memories table (without FK constraint)...")
            conn.execute(text("""
                CREATE TABLE user_memories (
                    id VARCHAR(36) PRIMARY KEY,
                    user_id VARCHAR(36),
                    fact TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()
            print("✓ user_memories table recreated successfully!")
            
        except Exception as e:
            print(f"✗ Error: {e}")

if __name__ == "__main__":
    fix_user_memories_table()
