# csv -> pandas data processing -> postgres
from utils import csv_to_pd, str_to_vec, setLogger, setLogHandler
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from utils import PostgresDataBase
from dotenv import load_dotenv
import os
from tables import Vectors
import subprocess
import sys

# --------------------------
# Docker Compose
# --------------------------
command = ["docker", "compose", "-f", "db/compose.yaml", "up", "-d", "long_term_db"]

try:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
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
today = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_filename = f'{today}.log'

handler = setLogHandler(log_dir=log_dir,
                        log_filename=log_filename,
                        mode='a',
                        setLevel='INFO')
logger.addHandler(handler)

# --------------------------
# Load csv to pandas, start db
# --------------------------
#! Change to correct path name if necessary
path = "./db/storage/output_2025_04_22_1.csv"

logger.info("==============")
logger.info(f"{path}")
logger.info(f"{datetime.now()}")
logger.info("==============")
logger.info("\n")


path = "./db/storage/output_2025-04-22_1.csv"
# either dataframe or TextFileReader (iteratable pandas chunks)
df = csv_to_pd(filepath=path,
               parse_dates=["timestamp"])


load_dotenv()
password = os.environ.get('POSTGRES_PASSWORD')
db_name = os.environ.get('LONG_TERM_DB')
port = os.environ.get('LONG_TERM_HOST_PORT')

db = PostgresDataBase(password=password,
                      db_name=db_name,
                      port=port)

# --------------------------
# Process and save data to db
# --------------------------
#! Change to correct table names/cols if necessary
trans_cols = ['timestamp', 'speaker', 'text']
vectors_cols = ['embedding_model', 'embedding_dim', 'embedding', 'index_type', 'index_measurement']

if isinstance(df, pd.DataFrame):
    # str to int
    df['embedding_dim'] = df['embedding_dim'].astype(int)
    # str to vector
    df['embedding'] = df['embedding'].apply(str_to_vec)

    # split df
    trans_df = df[trans_cols]
    vectors_df = df[vectors_cols]

    # transcriptions
    db.pandas_to_postgres(
        df=trans_df,
        table_name="transcriptions",
        logger=logger
    )
    # vectors
    db.pandas_to_postgres(
        df=vectors_df,
        table_name="vectors",
        logger=logger
    )
else: # iterator
    for i, chunk in enumerate(df):
        # str to int
        chunk['embedding_dim'] = chunk['embedding_dim'].astype(int)
        # str to vector
        chunk['embedding'] = chunk['embedding'].apply(str_to_vec)

        # split df
        trans_chunk = chunk[trans_cols]
        vectors_chunk = chunk[vectors_cols]

        logger.info(f"Chunk {i}: ")
        try:
            # transcriptions
            db.pandas_to_postgres(
                df=trans_chunk,
                table_name="transcriptions",
                logger=logger
            )
            # vectors
            db.pandas_to_postgres(
                df=vectors_chunk,
                table_name="vectors",
                logger=logger
            )
        except Exception as e:
            print(e)
            continue

# --------------------------
# Stop docker compose
# --------------------------
try:
    command = ["docker", "compose", "-f", "db/compose.yaml", "down"]
    subprocess.run(command, check=True)
    print("Docker Compose stopped successfully.")
except Exception as e:
    print(f"Error stopping Docker Compose: {e}")
    raise