from utils import (
    PostgresDataBase, 
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

#* This is for creating backup of short_term db (postgres -> csv)

#! Change table name as needed
tablename = "transcriptionsvectors"
true_options = ["yes", "y"]
false_options = ["no", "n"]

# --------------------------
# Create directory
# --------------------------
#! Change path as needed
# TODO: use program session env for name
copy_path = f'./db/backups/{tablename}_{today_str}'

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
#! Change file name as needed

# Export
db.postgres_to_csv(table_name=tablename, output_path=copy_name, compress=compress)







# TODO: move this logic to main
# Clean table (delete all rows)
# Note: make sure you have a copy of the data before deleting
clean = input_to_bool(question=f"Delete all rows from {tablename} table (y/n): ", true_options=true_options, false_options=false_options)
if clean:
    clean_table(db=db, tablename=tablename, truncate=True)



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