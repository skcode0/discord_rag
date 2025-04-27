from utils_async import (
    AsyncPostgresDataBase, 
    clean_table, 
    input_to_bool,
    is_valid_windows_name,
    windows_filename_validity_message,
    setLogger, 
    setLogHandler
    )
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from dotenv import load_dotenv
import os

#* This is for creating backup (postgres -> csv/gz)

#! Change table name as needed
tablename = "transcriptionsvectors"
true_options = ["yes", "y"]
false_options = ["no", "n"]

# load most recent program session name
load_dotenv(override=True)

#! Add/Modify env variables as needed
pg_username = os.environ.get('POSTGRESS_USER')
pg_password = os.environ.get('POSTGRES_PASSWORD')
short_db_name = os.environ.get('SHORT_TERM_DB')
short_port = os.environ.get('SHORT_TERM_HOST_PORT')
recent_sess = os.environ.get('PROGRAM_SESSION')

# --------------------------
# Create directory
# --------------------------
#! Change path as needed
copy_path = f'./db/backups/{tablename}_{recent_sess}'

# create dir (also parents) if none exists
Path(copy_path).mkdir(parents=True, exist_ok=True)


# --------------------------
# Docker Compose
# --------------------------
command = ["docker", "compose", "-f", "db/compose.yaml", "up", "-d", "short_term_db"]

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
today = datetime.now()
today_str = today.strftime("%Y-%m-%d")

logger = setLogger(setLevel = 'INFO')
log_filename = f'{today_str}.log'

handler = setLogHandler(log_dir=copy_path,
                        log_filename=log_filename,
                        mode='a',
                        setLevel='INFO')
logger.addHandler(handler)

logger.info(f"{today}\n")


# --------------------------
# Create copy
# --------------------------
# get valid file name
full_path = "copy.csv"
compress = False
while True:
    file_name = input("Give valid backup file name without extension: ")
    while not is_valid_windows_name(file_name):
        file_name = input("Give valid backup file name without extension: ")

    # compress or not
    compress = input_to_bool(question="Compress file? (y/n)", true_options=true_options, false_options=false_options)
    if compress:
        file_name += ".gz"
    else:
        file_name += ".csv"

    full_path = Path(copy_path) / file_name
    # if unique file path, stop loop
    if not full_path.exists():
        break

async def main():
    db = AsyncPostgresDataBase(password=pg_password,
                        db_name=short_db_name,
                        port=short_port,
                        hide_parameters=True)

    # Export
    await db.postgres_to_csv(table_name=tablename, output_path=full_path)
    # db backup
    # document: https://www.postgresql.org/docs/current/app-pgdump.html
    # time measurements: https://dan.langille.org/2013/06/10/using-compression-with-postgresqls-pg_dump/
    #TODO
    await db.dump_postgres()


    # Clean table (delete all rows)
    # Note: make sure you have a copy of the data before deleting
    clean = input_to_bool(question=f"Delete all rows from {tablename} table? Make sure you have a backup. (y/n): ", true_options=true_options, false_options=false_options)
    if clean:
        await clean_table(db=db, tablename=tablename, truncate=True)

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