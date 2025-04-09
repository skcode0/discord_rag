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

# TODO: start docker compose
# --------------------------
# Docker Compose
# --------------------------
command = ["docker", "compose", "-f", "db/compose.yaml", "up", "-d" "long_term_db"]

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
# csv -> pandas data processing -> postgres
# --------------------------
#! Change to correct path name if necessary
path = "./db/storage/sample.csv"

logger.info("==============")
logger.info(f"{path}")
logger.info(f"{datetime.now()}")
logger.info("==============")
logger.info("\n")

df = csv_to_pd(filepath=path,
               parse_date=["timestamp"])

# TODO: clean data (to vector, change to datetime)

load_dotenv()
password = os.environ.get('POSTGRES_PASSWORD')
db_name = os.environ.get('LONG_TERM_DB')
port = os.environ.get('LONG_TERM_HOST_PORT')

db = PostgresDataBase(password=password,
                      db_name=db_name,
                      port=port)

#! Change to correct table name if necessary
table_name = "vectors"
if isinstance(df, pd.DataFrame):
    db.pandas_to_postgres(
        df=df,
        table_name=table_name,
        logger=logger
    )
else: # iterator
    for i, chunk in enumerate(df):
        logger.info(f"Chunk {i}: ")
        try:
            db.pandas_to_postgres(
                df=chunk,
                table_name=table_name,
                logger=logger
            )
        except Exception as e:
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