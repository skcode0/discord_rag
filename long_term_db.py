# csv -> pandas data processing -> postgres
from utils import csv_to_pd, str_to_vec
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from utils import PostgresDataBase
from dotenv import load_dotenv
import os
from tables import Vectors

# --------------------------
# Set up logging
# ref: https://www.youtube.com/watch?v=urrfJgHwIJA
# --------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

log_dir = './db/storage/long_term_logs'
log_dir.mkdir(parents=True, exist_ok=True)

today = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_filename = f'{today}.log'

log_fullpath = Path(log_dir) / log_filename
handler = logging.FileHandler(log_fullpath, mode="a")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

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