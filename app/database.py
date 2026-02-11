from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# OPTIMIZED: Add connection pooling and performance settings
engine = create_engine(
    DATABASE_URL,
    pool_size=10,              # Maintain 10 connections in pool
    max_overflow=20,           # Allow up to 20 additional connections
    pool_pre_ping=True,        # Verify connections before using
    pool_recycle=3600,         # Recycle connections after 1 hour
    echo=False,                # Disable SQL logging for performance
    connect_args={
        "connect_timeout": 5,  # 5 second connection timeout
        "options": "-c statement_timeout=10000"  # 10 second query timeout
    } if "postgresql" in DATABASE_URL else {}
)

SessionLocal = sessionmaker(
    autocommit=False, 
    autoflush=False,  # Disable autoflush for better performance
    bind=engine
)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
