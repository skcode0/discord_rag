from sqlalchemy import Index, ForeignKey, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped, relationship
from datetime import datetime
from pgvector.sqlalchemy import Vector
from dotenv import load_dotenv
import os

load_dotenv()

embedding_dim = int(os.environ.get('EMBEDDING_DIM'))
embedding_dim=1024

# ref: https://www.youtube.com/watch?v=iwENqqgxm-g&list=PLKm_OLZcymWhtiM-0oQE2ABrrbgsndsn0&index=10

class Base(DeclarativeBase):
    pass

class Transcriptions(Base):
    __tablename__ = "transcriptions"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime]
    speaker: Mapped[str]
    text: Mapped[str]
    vector: Mapped["Vectors"] = relationship(back_populates="transcription", uselist=False, cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint(
            "timestamp",
            "speaker",
            "text",
            name="uq_transcriptions_timestamp_speaker_text"
        ),
    )


class Vectors(Base):
    # Reference: https://www.youtube.com/watch?v=iwENqqgxm-g&list=PLKm_OLZcymWhtiM-0oQE2ABrrbgsndsn0
    __tablename__ = "vectors"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    embedding_model: Mapped[str]
    embedding_dim: Mapped[int]
    embedding = mapped_column(Vector(embedding_dim))
    index_type: Mapped[str] = mapped_column(nullable=True) # ex. hnsw, ivf, streamingdiskann
    index_measurement: Mapped[str] = mapped_column(nullable=True) # ex. vector_cosine_ops, vector_l2_ops, vector_ip_ops, etc.
    transcription_id: Mapped[int] = mapped_column(ForeignKey("transcriptions.id"))
    transcription: Mapped["Transcriptions"] = relationship(back_populates="vector")

    # Reference: https://www.youtube.com/watch?v=WsDVBEmTlaI&list=PLKm_OLZcymWhtiM-0oQE2ABrrbgsndsn0&index=15
    # StreamingDiskAnn index (https://github.com/timescale/pgvectorscale/blob/main/README.md)
    __table_args__ = (
        Index(
            "embedding_idx",
            "embedding",
            postgresql_using="diskann",
            # postgresql_with={

            # } # index build parameters,
            postgresql_ops={"embedding": "vector_cosine_ops"}, # cosine similarity
        ),
    )


class CombinedBase(DeclarativeBase):
    pass

# combined table: transcriptions + vectors
class TranscriptionsVectors(CombinedBase):
    __tablename__ = "transcriptionsVectors"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime]
    speaker: Mapped[str]
    text: Mapped[str]

    embedding_model: Mapped[str]
    embedding_dim: Mapped[int]
    embedding = mapped_column(Vector(embedding_dim))
    index_type: Mapped[str] = mapped_column(nullable=True) # ex. hnsw, ivf, streamingdiskann
    index_measurement: Mapped[str] = mapped_column(nullable=True) # ex. vector_cosine_ops, vector_l2_ops, vector_ip_ops, etc.

    __table_args__ = (
        Index(
            "embedding_idx",
            "embedding",
            postgresql_using="diskann",
            # postgresql_with={

            # } # index build parameters,
            postgresql_ops={"embedding": "vector_cosine_ops"} # cosine similarity
        ),
    )
