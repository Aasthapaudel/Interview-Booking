from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.sql import func

from app.db.database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    strategy = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=func.now())


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(String, primary_key=True)
    doc_id = Column(String, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)


class Booking(Base):
    __tablename__ = "bookings"

    id = Column(String, primary_key=True)
    session_id = Column(String, nullable=False)
    name = Column(String)
    email = Column(String)
    date = Column(String)
    time = Column(String)
    created_at = Column(DateTime, server_default=func.now())
