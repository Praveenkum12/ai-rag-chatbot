"""
Quick test to verify UserMemory table and functionality
"""
import sys
sys.path.insert(0, 'd:/ai-rag-chatbot')

try:
    from app.database import SessionLocal
    from app.models import UserMemory
    
    db = SessionLocal()
    
    # Try to query the table
    try:
        count = db.query(UserMemory).count()
        print(f"✓ UserMemory table exists! Current memories: {count}")
        
        # Show recent memories
        memories = db.query(UserMemory).limit(5).all()
        for m in memories:
            print(f"  - [{m.user_id}] {m.fact}")
            
    except Exception as e:
        print(f"✗ Error accessing UserMemory table: {e}")
        print("\nAttempting to create table...")
        
        from app.models import Base
        from app.database import engine
        Base.metadata.create_all(bind=engine)
        print("✓ Tables created!")
        
    db.close()
    
except Exception as e:
    print(f"✗ Fatal error: {e}")
    import traceback
    traceback.print_exc()
