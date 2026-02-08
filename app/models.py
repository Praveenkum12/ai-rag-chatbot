from sqlalchemy import Column, String, DateTime, ForeignKey, Text, Integer, JSON
from sqlalchemy.dialects.postgresql import UUID
from .database import Base
import uuid
from datetime import datetime

class User(Base):
    __tablename__ = "users"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    full_name = Column(String(255), nullable=True)
    email = Column(String(255), unique=True, index=True)
    hashed_password = Column(String(255))
    phone = Column(String(20), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True) # Linked to User
    session_id = Column(String(255), nullable=True)
    title = Column(String(255), default="New Conversation")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# CREATE TABLE conversations (
#     id VARCHAR(36) NOT NULL,           -- Unique UUID (e.g. 3111a276-...)
#     user_id VARCHAR(255) DEFAULT 'guest', 
#     session_id VARCHAR(255),           -- Currently NULL (reserved for future use)
#     title VARCHAR(255),                -- The sidebar name (first 50 chars of chat)
#     created_at DATETIME,               -- When the chat started
#     updated_at DATETIME,               -- When the last message was sent
#     PRIMARY KEY (id)
# );

class Message(Base):
    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=False)
    role = Column(String(50)) # 'user' or 'assistant'
    content = Column(Text)
    token_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


# CREATE TABLE messages (
#     id VARCHAR(36) NOT NULL,           -- Unique UUID for this specific message
#     conversation_id VARCHAR(36) NOT NULL, -- Links this message to a Conversation
#     role VARCHAR(50),                  -- 'user' or 'assistant'
#     content TEXT,                      -- The actual text of the message
#     token_count INTEGER,               -- Calculated tokens (tiktoken)
#     created_at DATETIME,               -- Precise timestamp of the message
#     PRIMARY KEY (id),
#     FOREIGN KEY(conversation_id) REFERENCES conversations (id)
# );
class UserMemory(Base):
    __tablename__ = "user_memories"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=True)  # No FK constraint to allow 'guest'
    fact = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
