from sqlalchemy import Index, UniqueConstraint, BigInteger, DateTime
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped
from datetime import datetime
from pgvector.sqlalchemy import Vector
from dotenv import load_dotenv
import os

load_dotenv(override=True)

embedding_dim = int(os.environ.get('EMBEDDING_DIM'))

# ref: https://www.youtube.com/watch?v=iwENqqgxm-g&list=PLKm_OLZcymWhtiM-0oQE2ABrrbgsndsn0&index=10

class CombinedBase(DeclarativeBase):
    pass

# combined table: transcriptions + vectors
class TranscriptionsVectors(CombinedBase):
    __tablename__ = "transcriptionsvectors"
    trans_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True, autoincrement=False) # timezone aware
    speaker: Mapped[str]
    text: Mapped[str]

    vec_id: Mapped[int] = mapped_column(BigInteger)
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
        UniqueConstraint(
            "timestamp",
            "speaker",
            "text",
            name="uq_transcriptions_timestamp_speaker_text"
        ),
    )
