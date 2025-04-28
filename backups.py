from utils_async import (
    AsyncPostgresDataBase, 
    clean_table, 
    input_to_bool,
    is_valid_windows_name,
    setLogger, 
    setLogHandler,
    close_docker_compose
    )
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from dotenv import load_dotenv
import os
import asyncio

#* This is for creating backup (postgres -> csv)

#! Change table/db name as needed
tablename = "transcriptionsvectors"
db_name = "short_term_db"
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
copy_path = f'./db/backups/{recent_sess}'

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


# --------------------------
# Create copy
# --------------------------
# get valid file name
file_name = "copy"
while True:
    file_name = input("Give valid backup file name without file extension: ")
    while not is_valid_windows_name(file_name):
        file_name = input("Give valid backup file name without file extension: ")

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

    tasks = [
        # export as csv
        db.postgres_to_csv(table_name=tablename, output_path=full_path),
        # db backup
        # document: https://www.postgresql.org/docs/current/app-pgdump.html
        # time measurements: https://dan.langille.org/2013/06/10/using-compression-with-postgresqls-pg_dump/
        db.dump_postgres(backup_path=copy_path, 
                         database_name=db_name, 
                         F="c")
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)
    errors = []

    logger.info(f"{today}\n")

    for result in results:
        if isinstance(result, Exception): # error
            logger.error(result)
            errors.append(result)
        else:
            logger.info(f"Backup successful in {copy_path}.\n")
        
    if errors:
        raise ExceptionGroup(errors)


    # Clean table (delete all rows)
    # Note: make sure you have a copy of the data before deleting
    clean = input_to_bool(question=f"Delete all rows from {tablename} table? Make sure you have a backup. (y/n): ", true_options=true_options, false_options=false_options)
    if clean:
        await clean_table(db=db, tablename=tablename, truncate=True)

    # --------------------------
    # Stop docker compose
    # --------------------------
    try:
        print("Everything added successfully. Press Ctrl+c to shut down.")
        await asyncio.Event().wait()
    except (asyncio.CancelledError, KeyboardInterrupt) as e:
        print("Shutting down...")
    finally:
        await close_docker_compose(compose_path="./db/compose.yaml", down=False)


asyncio.run(main())