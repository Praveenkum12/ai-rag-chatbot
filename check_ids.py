from app.database import SessionLocal
from app.models import Conversation

db = SessionLocal()
convs = db.query(Conversation).all()
for c in convs:
    print(f"ID: '{c.id}' | Title: '{c.title}'")
db.close()
