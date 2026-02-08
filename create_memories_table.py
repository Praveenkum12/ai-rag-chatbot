"""
Migration script to create the user_memories table
"""
from app.database import engine
from sqlalchemy import text

def create_user_memories_table():
    with engine.connect() as conn:
        # Check if table exists
        result = conn.execute(text("""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_schema = DATABASE() 
            AND table_name = 'user_memories'
        """))
        
        if result.scalar() == 0:
            print("Creating user_memories table...")
            conn.execute(text("""
                CREATE TABLE user_memories (
                    id VARCHAR(36) PRIMARY KEY,
                    user_id VARCHAR(36),
                    fact TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """))
            conn.commit()
            print("✓ user_memories table created successfully!")
        else:
            print("✓ user_memories table already exists")

if __name__ == "__main__":
    create_user_memories_table()
