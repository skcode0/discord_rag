import os
from util_funcs import make_pgdb
from sqlalchemy import create_engine, String, Column, Integer, DateTime, func, Index
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
from datetime import datetime
import subprocess

# --------------------------
# start docker compose command
# --------------------------
command = ["docker", "compose", "up", "-d"]

try:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    print("Docker Compose Output:\n", result.stdout)
except subprocess.CalledProcessError as e:
    print("Error running docker-compose:", e.stderr)

# --------------------------
# Connecting to database
# --------------------------
pg_username = os.environ.get('POSTGRESS_USER')
pg_password = os.environ.get('POSTGRES_PASSWORD')
db_name = os.environ.get('SHORT_TERM_DB')
port = os.environ.get('SHORT_TERM_HOST_PORT')

# creates db if there's none
url = make_pgdb(password=pg_password,
                db=db_name,
                port=port,
                add_vectors=True)

engine = create_engine(url, echo=False)
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()

# IMPORTANT: check and change value if necessary when you use different embedding model
embedding_dim = 1024

class Vectors(Base):
    __tablename__ = "vectors"
    id = Column(Integer, primary_key=True) # will auto increment
    time_spoken = Column(DateTime, nullable=False)
    speaker = Column(String, nullable=False)
    text = Column(String, nullable=False)
    embedding = Column(Vector(embedding_dim), nullable=False)

    # StreamingDiskAnn index
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

Base.metadata.create_all(engine) # prevents duplicate tables


today = datetime.today().strftime('%Y_%m_%d')
# for rows that were not sucessfully added to db for some reason (ex. duplicates, inserting null in non-nullable column, etc.)
not_added_path = "storage/" + f"rows_not_added_{today}.csv" 
# for storing all data in csv format
path = "storage/" + f"{today}"


# --------------------------
# Start discord, create embeddings, and add vectors to db
# --------------------------
#TODO embedding model (other option: bge-m3 567M)
# intfloat/multilingual-e5-large-instruct (560M)
embedding_model = None




