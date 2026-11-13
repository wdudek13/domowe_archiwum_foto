import datetime
from typing import List, Optional
from sqlalchemy import (
    BigInteger, Boolean, DateTime, Float, ForeignKey,
    Index, Integer, LargeBinary, String, Text
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class WindowsVolume(Base):
    __tablename__ = "windows_volumes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    volume_serial_number: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    volume_label: Mapped[str] = mapped_column(String(255), default="NO_NAME", nullable=False)
    file_system: Mapped[str] = mapped_column(String(32), default="NTFS", nullable=False)
    last_known_drive_letter: Mapped[str] = mapped_column(String(16), nullable=False)
    is_online: Mapped[bool] = mapped_column(Boolean, default=True, index=True, nullable=False)
    last_seen_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    media_items: Mapped[List["MediaItem"]] = relationship("MediaItem", back_populates="volume")


class MediaGroup(Base):
    __tablename__ = "media_groups"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    parent_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("media_groups.id", ondelete="CASCADE"), nullable=True, index=True
    )
    target_volume_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("windows_volumes.id", ondelete="SET NULL"), nullable=True
    )
    is_trashed: Mapped[bool] = mapped_column(Boolean, default=False, index=True, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    parent: Mapped[Optional["MediaGroup"]] = relationship("MediaGroup", remote_side=[id], back_populates="children")
    children: Mapped[List["MediaGroup"]] = relationship("MediaGroup", back_populates="parent", cascade="all, delete-orphan")
    media_items: Mapped[List["MediaItem"]] = relationship("MediaItem", back_populates="group")


class MediaItem(Base):
    __tablename__ = "media_items"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    volume_id: Mapped[int] = mapped_column(Integer, ForeignKey("windows_volumes.id", ondelete="RESTRICT"), nullable=False, index=True)
    relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    media_type: Mapped[str] = mapped_column(String(16), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(64), nullable=False)

    file_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    hash_algorithm: Mapped[str] = mapped_column(String(32), nullable=False)
    is_primary_duplicate: Mapped[bool] = mapped_column(Boolean, default=True, index=True, nullable=False)

    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    exif_date: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime, nullable=True)
    filename_date: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime, nullable=True)
    fs_ctime: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False)
    fs_mtime: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False)
    effective_date: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False, index=True)

    has_date_anomaly: Mapped[bool] = mapped_column(Boolean, default=False, index=True, nullable=False)
    date_anomaly_reason: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    is_corrupted: Mapped[bool] = mapped_column(Boolean, default=False, index=True, nullable=False)
    is_blurry: Mapped[bool] = mapped_column(Boolean, default=False, index=True, nullable=False)
    blur_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    thumbnail_relative_path: Mapped[str] = mapped_column(String(255), nullable=False)
    group_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("media_groups.id", ondelete="SET NULL"), nullable=True, index=True)

    is_trashed: Mapped[bool] = mapped_column(Boolean, default=False, index=True, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow, nullable=False
    )

    volume: Mapped["WindowsVolume"] = relationship("WindowsVolume", back_populates="media_items")
    group: Mapped[Optional["MediaGroup"]] = relationship("MediaGroup", back_populates="media_items")
    tags: Mapped[List["MediaTag"]] = relationship("MediaTag", back_populates="media_item", cascade="all, delete-orphan")
    embedding: Mapped[Optional["MediaEmbedding"]] = relationship("MediaEmbedding", back_populates="media_item", uselist=False, cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_media_composite_filter", "media_type", "effective_date"),
        Index("idx_volume_relative_unique", "volume_id", "relative_path", unique=True),
    )


class MediaTag(Base):
    __tablename__ = "media_tags"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    media_id: Mapped[str] = mapped_column(String(36), ForeignKey("media_items.id", ondelete="CASCADE"), nullable=False, index=True)
    tag: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    source_model: Mapped[str] = mapped_column(String(64), nullable=False)

    media_item: Mapped["MediaItem"] = relationship("MediaItem", back_populates="tags")


class MediaEmbedding(Base):
    __tablename__ = "media_embeddings"
    media_id: Mapped[str] = mapped_column(String(36), ForeignKey("media_items.id", ondelete="CASCADE"), primary_key=True)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    media_item: Mapped["MediaItem"] = relationship("MediaItem", back_populates="embedding")
