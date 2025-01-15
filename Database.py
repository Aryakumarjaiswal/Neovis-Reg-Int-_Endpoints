from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import bcrypt
from sqlalchemy import (
    Column, String, Integer, Enum, ForeignKey, Text, TIMESTAMP, func, CHAR
)
import uuid
from sqlalchemy.dialects.mysql import CHAR
DATABASE_URL = "mysql+pymysql://root:#1Krishna@localhost:3306/Conversations"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()




def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


class Session_Table(Base):
    __tablename__ = "Session_table_2"

    session_id = Column(CHAR(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    user_id = Column(String(50))
    user_type = Column(String(50),nullable=False, default="user")
    status = Column(Enum("active", "closed", "transferred"), nullable=False, default="active")
    started_at = Column(String(50),nullable=True)
    ended_at = Column(String(50),nullable=True)
    Duration=Column(String(50),nullable=True)
    # Relationships to other tables
    chats = relationship("Chat", back_populates="session")
    chat_transfers = relationship("ChatTransfer", back_populates="session")


class Chat(Base):
    __tablename__ = "Chat_table"

    chat_id = Column(Integer, primary_key=True,index=True,autoincrement=True)
    session_id = Column(String(225), ForeignKey("Session_table_2.session_id"))  # Fixed ForeignKey
    sender = Column(Enum("user", "bot", "agent"), nullable=False)
    message = Column(Text, nullable=False)
    sent_at = Column(String(50),nullable=True)
    status = Column(Enum("unread", "read"), default="read")

    # Relationship back to the Session
    session = relationship("Session_Table", back_populates="chats")


class ChatTransfer(Base):
    __tablename__ = "chat_transfer_table"

    transfer_id = Column(CHAR(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    session_id = Column(String(225), ForeignKey("Session_table_2.session_id"))  # Fixed ForeignKey
    transferred_by = Column(String(50))
    transfer_reason = Column(Text, nullable=True)
    transferred_at =Column(String(50),nullable=True)
    agent_id= Column(CHAR(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    # Relationship back to the Session
    session = relationship("Session_Table", back_populates="chat_transfers")


print("Tables to be created:")
#Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

print("Tables created successfully.")
