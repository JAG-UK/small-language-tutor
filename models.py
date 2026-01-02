from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class Conversation(Base):
    __tablename__ = 'conversations'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(200))
    language = Column(String(10))
    created_at = Column(DateTime, default=datetime.utcnow)
    messages = Column(JSON)  # Store messages as JSON array
    corrections = Column(JSON)  # Store corrections as JSON array

engine = create_engine('sqlite:///database.db', echo=False)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

