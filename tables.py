from sqlalchemy import Index
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped
from datetime import datetime
from pgvector.sqlalchemy import Vector
from dotenv import load_dotenv
import os

load_dotenv()

embedding_dim = int(os.environ.get('EMBEDDING_DIM'))


class Base(DeclarativeBase):
    pass

class Vectors(Base):
    # Reference: https://www.youtube.com/watch?v=iwENqqgxm-g&list=PLKm_OLZcymWhtiM-0oQE2ABrrbgsndsn0
    __tablename__ = "vectors"
    id: Mapped[int] = mapped_column(primary_key=True) # will auto increment
    timestamp: Mapped[datetime]
    speaker: Mapped[str]
    text: Mapped[str]
    embedding = mapped_column(Vector(embedding_dim))

    # Reference: https://www.youtube.com/watch?v=WsDVBEmTlaI&list=PLKm_OLZcymWhtiM-0oQE2ABrrbgsndsn0&index=15
    # StreamingDiskAnn index (https://github.com/timescale/pgvectorscale/blob/main/README.md)
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