from datetime import datetime, timezone
from functools import lru_cache
from typing import Generator
from urllib.parse import quote_plus

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

try:
    from .config import get_secret
except ImportError:
    from config import get_secret


class Base(DeclarativeBase):
    pass


class Case(Base):
    __tablename__ = "Cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    user_name: Mapped[str | None] = mapped_column(String(255))
    reason: Mapped[str] = mapped_column(String(500), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)
    vehicle_number: Mapped[str | None] = mapped_column(String(50))
    vehicle_category: Mapped[str | None] = mapped_column(String(100))
    latitude: Mapped[str | None] = mapped_column(String(100))
    longitude: Mapped[str | None] = mapped_column(String(100))
    timestamp: Mapped[str | None] = mapped_column(String(100))
    chat_history: Mapped[str | None] = mapped_column(Text)
    language: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    images: Mapped[list["CaseImage"]] = relationship(
        back_populates="case",
        cascade="all, delete-orphan",
    )


class CaseImage(Base):
    __tablename__ = "CaseImages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[int] = mapped_column(ForeignKey("Cases.id"), nullable=False, index=True)
    image_url: Mapped[str] = mapped_column(Text, nullable=False)
    original_filename: Mapped[str | None] = mapped_column(String(512))
    content_type: Mapped[str | None] = mapped_column(String(255))
    size_bytes: Mapped[int | None] = mapped_column(Integer)
    sha256: Mapped[str | None] = mapped_column(String(64))

    case: Mapped[Case] = relationship(back_populates="images")


class ChatSession(Base):
    __tablename__ = "ChatSessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    user_name: Mapped[str | None] = mapped_column(String(255))
    session_type: Mapped[str] = mapped_column(String(20), nullable=False, default="global")
    parent_session_id: Mapped[int | None] = mapped_column(Integer, nullable=True)  # FK to ChatSessions.id for branches
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )


class ChatMessage(Base):
    __tablename__ = "ChatMessages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("ChatSessions.id"), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(50), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    message_type: Mapped[str] = mapped_column(String(50), default="text", nullable=False)
    image_url: Mapped[str | None] = mapped_column(Text)  # Base64 data URL or URL to image for image-type messages
    language: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    session: Mapped[ChatSession] = relationship(back_populates="messages")


class UserLocation(Base):
    """Stores GPS pings from field officers for dashboard tracking."""
    __tablename__ = "UserLocations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    user_name: Mapped[str | None] = mapped_column(String(255))
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    accuracy: Mapped[float | None] = mapped_column(Float)  # metres, optional
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )


class OfficerDevice(Base):
    __tablename__ = "officer_devices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    officer_id: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    fcm_token: Mapped[str | None] = mapped_column(String(500))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


def ensure_officer_devices_table(session: Session) -> None:
    session.execute(
        text(
            """
            IF OBJECT_ID('dbo.officer_devices', 'U') IS NULL
            BEGIN
                CREATE TABLE dbo.officer_devices (
                    id INT IDENTITY PRIMARY KEY,
                    officer_id NVARCHAR(200) NOT NULL,
                    fcm_token NVARCHAR(500) NULL,
                    updated_at DATETIME2 NOT NULL DEFAULT GETUTCDATE()
                );
                CREATE INDEX ix_officer_devices_officer_id ON dbo.officer_devices (officer_id);
            END
            """
        )
    )
    session.commit()


def _database_url() -> str:
    try:
        database_url = get_secret("database-url")
    except Exception:
        database_url = None

    if database_url:
        return database_url

    odbc_connection_string = get_secret("azure-sql-connection-string")
    if odbc_connection_string:
        if "timeout=" not in odbc_connection_string.lower():
            odbc_connection_string = f"{odbc_connection_string.rstrip(';')};Connection Timeout=120;Login Timeout=120"

        return f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc_connection_string)}"

    raise RuntimeError("DATABASE_URL or AZURE_SQL_CONNECTION_STRING is not set.")


@lru_cache()
def get_engine():
    return create_engine(_database_url(), pool_pre_ping=True)


@lru_cache()
def get_sessionmaker():
    return sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)


def get_db_session() -> Generator[Session, None, None]:
    session = get_sessionmaker()()
    try:
        yield session
    finally:
        session.close()


def create_database_schema() -> None:
    Base.metadata.create_all(bind=get_engine())


@lru_cache()
def ensure_chat_schema() -> None:
    return None


def ensure_cases_user_name_column(session: Session) -> None:
    session.execute(
        text(
            """
            IF COL_LENGTH('dbo.Cases', 'user_name') IS NULL
            BEGIN
                ALTER TABLE dbo.Cases ADD user_name NVARCHAR(255) NULL
            END
            """
        )
    )


def ensure_chat_message_image_url_column(session: Session) -> None:
    """Ensure ChatMessages table has image_url column for storing image URLs/base64"""
    session.execute(
        text(
            """
            IF COL_LENGTH('dbo.ChatMessages', 'image_url') IS NULL
            BEGIN
                ALTER TABLE dbo.ChatMessages ADD image_url NVARCHAR(MAX) NULL
            END
            ELSE
            BEGIN
                ALTER TABLE dbo.ChatMessages ALTER COLUMN image_url NVARCHAR(MAX) NULL
            END
            """
        )
    )
    session.commit()


def ensure_case_images_image_url_column_type(session: Session) -> None:
    """Ensure CaseImages table's image_url column has NVARCHAR(MAX) type for base64 strings"""
    session.execute(
        text(
            """
            IF COL_LENGTH('dbo.CaseImages', 'image_url') IS NOT NULL
            BEGIN
                ALTER TABLE dbo.CaseImages ALTER COLUMN image_url NVARCHAR(MAX) NOT NULL
            END
            """
        )
    )
    session.commit()


def ensure_cases_vehicle_columns(session: Session) -> None:
    """Add vehicle_number and vehicle_category columns to Cases table if missing."""
    session.execute(
        text(
            """
            IF COL_LENGTH('dbo.Cases', 'vehicle_number') IS NULL
            BEGIN
                ALTER TABLE dbo.Cases ADD vehicle_number NVARCHAR(50) NULL
            END
            """
        )
    )
    session.execute(
        text(
            """
            IF COL_LENGTH('dbo.Cases', 'vehicle_category') IS NULL
            BEGIN
                ALTER TABLE dbo.Cases ADD vehicle_category NVARCHAR(100) NULL
            END
            """
        )
    )
    session.commit()


def ensure_user_location_table(session: Session) -> None:
    """Create the UserLocations table and indexes if they do not yet exist."""
    session.execute(
        text(
            """
            IF NOT EXISTS (
                SELECT 1 FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_NAME = 'UserLocations'
            )
            BEGIN
                CREATE TABLE dbo.UserLocations (
                    id           INT IDENTITY(1,1) PRIMARY KEY,
                    user_id      NVARCHAR(255)    NOT NULL,
                    user_name    NVARCHAR(255)    NULL,
                    latitude     FLOAT            NOT NULL,
                    longitude    FLOAT            NOT NULL,
                    accuracy     FLOAT            NULL,
                    recorded_at  DATETIMEOFFSET   NOT NULL DEFAULT SYSDATETIMEOFFSET()
                );
                CREATE INDEX IX_UserLocations_user_id     ON dbo.UserLocations (user_id);
                CREATE INDEX IX_UserLocations_recorded_at ON dbo.UserLocations (recorded_at);
            END
            """
        )
    )
    session.commit()


def ensure_chat_session_branch_columns(session: Session) -> None:
    """Migrate ChatSessions table to support multiple sessions per user (branch chat).
    
    - Drops the unique constraint on user_id if present (idempotent).
    - Adds session_type and parent_session_id columns if missing.
    """
    # Drop the unique index on user_id if it still exists (was created before branching was supported)
    session.execute(
        text(
            """
            IF EXISTS (
                SELECT 1 FROM sys.indexes
                WHERE object_id = OBJECT_ID('dbo.ChatSessions')
                  AND name = 'ix_chatsessions_user_id'
                  AND is_unique = 1
            )
            BEGIN
                DROP INDEX ix_chatsessions_user_id ON dbo.ChatSessions;
                CREATE INDEX ix_chatsessions_user_id ON dbo.ChatSessions (user_id);
            END
            """
        )
    )
    # Add user_name column
    session.execute(
        text(
            """
            IF COL_LENGTH('dbo.ChatSessions', 'user_name') IS NULL
            BEGIN
                ALTER TABLE dbo.ChatSessions ADD user_name NVARCHAR(255) NULL
            END
            """
        )
    )
    # Add session_type column
    session.execute(
        text(
            """
            IF COL_LENGTH('dbo.ChatSessions', 'session_type') IS NULL
            BEGIN
                ALTER TABLE dbo.ChatSessions ADD session_type NVARCHAR(20) NOT NULL DEFAULT 'global'
            END
            """
        )
    )
    # Add parent_session_id column
    session.execute(
        text(
            """
            IF COL_LENGTH('dbo.ChatSessions', 'parent_session_id') IS NULL
            BEGIN
                ALTER TABLE dbo.ChatSessions ADD parent_session_id INT NULL
            END
            """
        )
    )
    session.commit()
