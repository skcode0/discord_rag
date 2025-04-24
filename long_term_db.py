from utils import csv_to_pd, str_to_vec, setLogger, setLogHandler
import pandas as pd
from datetime import datetime, timezone
from utils import PostgresDataBase
from dotenv import load_dotenv
import os
import subprocess
import sys
from tables import Base
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector

# --------------------------
# Docker Compose
# --------------------------
command = ["docker", "compose", "-f", "db/compose.yaml", "up", "-d", "long_term_db"]

try:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    if result.stdout.strip():
        print("Docker Compose Output:\n", result.stdout)
except subprocess.CalledProcessError as e:
    print("Error running docker-compose:", e.stderr)
    sys.exit(1)

# --------------------------
# Set up logging
# ref: https://www.youtube.com/watch?v=urrfJgHwIJA
# --------------------------
logger = setLogger(setLevel = 'INFO')

log_dir = './db/storage/long_term_logs'
today = datetime.now().strftime('%Y-%m-%d')
log_filename = f'{today}.log'

handler = setLogHandler(log_dir=log_dir,
                        log_filename=log_filename,
                        mode='a',
                        setLevel='INFO')
logger.addHandler(handler)

# --------------------------
# Load csv to pandas, create db/tables if none exists
# --------------------------
#! Change to correct path name if necessary
path = "./db/storage/output_2025_04_22_1.csv"

logger.info(f"{path}")
logger.info(f"{datetime.now()}\n")

path = "./db/storage/output_2025-04-22_1.csv"
# either dataframe or TextFileReader (iteratable pandas chunks)
#! Change chunksize if necessary
chunksize = None
df = csv_to_pd(filepath=path,
               parse_dates=["timestamp"],
               chunksize=chunksize)


load_dotenv(override=True)
password = os.environ.get('POSTGRES_PASSWORD')
db_name = os.environ.get('LONG_TERM_DB')
port = os.environ.get('LONG_TERM_HOST_PORT')
embedding_dim = os.environ.get('EMBEDDING_DIM')
embedding_dim = int(embedding_dim)

db = PostgresDataBase(password=password,
                      db_name=db_name,
                      port=port,
                      echo=False,
                      hide_parameters=True)

url = db.make_db()
db.enable_vectors()

try:
    # create table(s)
    Base.metadata.create_all(db.engine) # prevents duplicate tables
    Session = sessionmaker(bind=db.engine)
except Exception as e:
    print(e)

# --------------------------
# Process and save data to db
# --------------------------
#! Change to correct table names/cols if necessary
trans_cols = ['id', 'timestamp', 'timezone', 'speaker', 'text'] # id instead of trans_id because col name would have already changed to id.
vectors_cols = ['vec_id', 'id', 'embedding_model', 'embedding_dim', 'embedding', 'index_type', 'index_measurement']

if isinstance(df, pd.DataFrame):
    # str to int
    df['embedding_dim'] = df['embedding_dim'].astype(int)
    # str to vector
    df['embedding'] = df['embedding'].apply(str_to_vec, args=(False,))
    # change col name
    df = df.rename(columns={'trans_id': 'id'})

    # split df
    trans_df = df[trans_cols]
    vectors_df = df[vectors_cols]

    # vectors fk
    vectors_df = vectors_df.rename(columns={'id': 'transcription_id'})
    # vectors pk
    vectors_df = vectors_df.rename(columns={'vec_id': 'id'})

    try:
        # transcriptions
        db.pandas_to_postgres(
            df=trans_df,
            table_name="transcriptions",
            logger=logger
        )
    except Exception as e:
        print(e)
    
    try:
        # vectors
        db.pandas_to_postgres(
            df=vectors_df,
            table_name="vectors",
            logger=logger,
            dtype = {"embedding": Vector(embedding_dim)}
        )
        print("")
    except Exception as e:
        print(e)
else: # iterator
    for i, chunk in enumerate(df):
        # str to int
        chunk['embedding_dim'] = chunk['embedding_dim'].astype(int)
        # str to vector
        chunk['embedding'] = chunk['embedding'].apply(str_to_vec, args=(True,))
        # change col name
        chunk = chunk.rename(columns={'trans_id': 'id'})

        # split df
        trans_chunk = chunk[trans_cols]
        vectors_chunk = chunk[vectors_cols]

        # vectors fk
        vectors_chunk = vectors_chunk.rename(columns={'id': 'transcription_id'})
        # vectors pk
        vectors_chunk = vectors_chunk.rename(columns={'vec_id': 'id'})

        logger.info(f"Chunk {i} ({len(chunk)} rows): ")
        try:
            # transcriptions
            db.pandas_to_postgres(
                df=trans_chunk,
                table_name="transcriptions",
                logger=logger
            )
        except Exception as e:
            print(e)

        try:
            # vectors
            db.pandas_to_postgres(
                df=vectors_chunk,
                table_name="vectors",
                logger=logger,
                dtype = {"embedding": Vector(embedding_dim)}
            )
        except Exception as e:
            print(e)

# --------------------------
# Stop docker compose
# --------------------------
try:
    while True:
        pass
except KeyboardInterrupt:
    try:
        command = ["docker", "compose", "-f", "db/compose.yaml", "stop"] # 'down' deletes container and network
        subprocess.run(command, check=True)
        print("Docker Compose stopped successfully.")
    except Exception as e:
        print(f"Error stopping Docker Compose: {e}")
        raise