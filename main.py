import os
from utils import (
    PostgresDataBase, 
    create_program_session_dir, 
    name_and_write_to_csv,
    write_to_csv, 
    clean_table, 
    close_docker_compose, 
    get_detailed_instruct, 
    create_embedding,
    input_to_bool,
    is_valid_windows_name,
    windows_filename_validity_message,
    )
from pathlib import Path 
from sqlalchemy import create_engine, Index
from sqlalchemy.orm import sessionmaker, mapped_column, Mapped
from pgvector.sqlalchemy import Vector
from datetime import datetime, timezone
import subprocess
import discord
from discord.ext import commands
from dotenv import load_dotenv
import sys
from tables import CombinedBase, Base, Transcriptions, Vectors, TranscriptionsVectors
from sonyflake import SonyFlake

#* Note: This mostly uses synchronous functions. Async version of the code is in 'main_async.py'.

# --------------------------
# start Docker Compose command for DBs (only short term)
# --------------------------
command = ["docker", "compose", "-f", "db/compose.yaml", "up", "-d"]

try:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    if result.stdout.strip():
        print("Docker Compose Output:\n", result.stdout)
except subprocess.CalledProcessError as e:
    print("Error running docker-compose:", e.stderr)
    sys.exit(1)

# --------------------------
# Load in environment variables
# --------------------------
# postgres
load_dotenv(override=True)
pg_username = os.environ.get('POSTGRESS_USER')
pg_password = os.environ.get('POSTGRES_PASSWORD')
short_db_name = os.environ.get('SHORT_TERM_DB')
short_port = os.environ.get('SHORT_TERM_HOST_PORT')
long_db_name = os.environ.get('LONG_TERM_DB')
long_port = os.environ.get('LONG_TERM_HOST_PORT')

# discord
discord_token = os.environ.get('DISCORD_TOKEN')

# HF access token
hf_token = os.environ.get('hf_token')

# llm model
llm_model = os.environ.get('LLM_MODEL')

# embedding model
embedding_dim = int(os.environ.get('EMBEDDING_DIM'))
embedding_model = os.environ.get('EMBEDDING_MODEL')

# program session folder
program_session = os.environ.get('PROGRAM_SESSION')


# --------------------------
# For .csv data storage
# --------------------------
program_session = create_program_session_dir() # create session folder

storage_path = './db/storage'

# For rows that were not successfully added to db for some reason (ex. duplicates, inserting null in non-nullable column, etc.)
not_added_file_name = "not_added.csv"
# for recording all data
all_file_name = "output.csv"

true_options = ["yes", "y"]
false_options = ["no", "n"]
print("\n--------------------------------------")
print("Following questions are for writing data (in csv) that has failed to save in database.")
print("--------------------------------------")
add_date = input_to_bool(question="Add date? (y/n):", true_options=true_options, false_options=false_options)

auto_increment = input_to_bool(question="Add auto increment for file numbering? (y/n): ", true_options=true_options, false_options=false_options)

not_added_csv_path = name_and_write_to_csv(file_path=storage_path,
                                           file_name=not_added_file_name,
                                  session_name=program_session,
                                  add_date = add_date,
                                  auto_increment=auto_increment)

# This is for saving data in csv regardless of whether data was successfully added to database or not. In other words, this is simply a backup save.
print("\n--------------------------------------")
print("Following questions are for writing all data to csv (back-up file).")
print("--------------------------------------")
add_date = input_to_bool(question="Add date? (y/n):", true_options=true_options, false_options=false_options)

auto_increment = input_to_bool(question="Add auto increment for file numbering? (y/n): ", true_options=true_options, false_options=false_options)

all_records_csv_path = name_and_write_to_csv(file_path=storage_path,
                                             file_name=all_file_name,
                                  session_name=program_session,
                                  add_date = add_date,
                                  auto_increment=auto_increment)

# --------------------------
# Create database (+ postgres extensions) and table if not present
# --------------------------
# short-term long-term db
db = PostgresDataBase(password=pg_password,
                      db_name=short_db_name,
                      port=short_port,
                      hide_parameters=True)

url = db.make_db()
db.enable_vectors()

# create table(s)
CombinedBase.metadata.create_all(db.engine) # prevents duplicate tables
Session = sessionmaker(bind=db.engine)

# pk generation 
#! Change start_date, machine_id if necessary
start_time = datetime(2025,1,1,tzinfo=timezone.utc)
sf = SonyFlake(start_time=start_time, machine_id=lambda: 1)

#! DUMMY DATA
# https://weaviate.io/blog/vector-embeddings-explained
emb = [
        ["cat", [1.5, -0.4, 7.2, 19.6, 20.2]],
        ["dog", [1.7, -0.3, 6.9, 19.1, 21.1]],
        ["apple", [-5.2, 3.1, 0.2, 8.1, 3.5]],
        ["strawberry", [-4.9, 3.6, 0.9, 7.8, 3.6]],
        ["building",[60.1, -60.3, 10, -12.3, 9.2]],
        ["car",[81.6, -72.1, 16, -20.2, 102]]
]

snowflake_id = [7320991239237537792, 7320991239237537793, 7320991239237537794, 7320991239237537795, 7320991239237537796, 7320991239237537797]

try:
    for i in range(len(emb)):
        data = {
                    # Transcriptions
                    "trans_id": snowflake_id[i],
                    "timestamp": datetime.now(),
                    "timezone": "CT", #* change accordinly,
                    "speaker": f"user_{i}",
                    "text": emb[i][0],

                    # Vectors
                    "vec_id": sf.next_id(),
                    "embedding_model": embedding_model,
                    "embedding_dim": embedding_dim,
                    "embedding": emb[i][1],
                    "index_type": "StreamingDiskAnn", #* change accordingly
                    "index_measurement": "vector_cosine_ops", #* change accordingly
                }
        
        db.add_record(table=TranscriptionsVectors,data=data)
except:
    pass
#! DUMMY DATA

# --------------------------
# Store Discord messages as embeddings (+ csv files) and call llm with rag to answer user inputs
# --------------------------
class MyBot(commands.Bot):
    async def on_ready(self):
        print(f"Logged on as {bot.user}!")

    async def on_message(self, message):
        # ignore replaying to itself
        if message.author == bot.user:
            return
        
        # Create embedding
        # Note: Some embedding models like 'intfloat/multilingual-e5-large-instruct' require instructions to be added to query. Documents don't need instructions.
        task = "Given user's message query, retrieve relevant messages that answer the query."
        # instruct_query = get_detailed_instruct(query=message.content,
        #                                         task_description=task)
        # # for querying
        # instruct_embedding = create_embedding(model_name=embedding_model,
        #                                       input=instruct_query)
        

        # for storage
        # embedding_vector = create_embedding(model_name=embedding_model,
        #                                     input=f"{message.created_at.strftime("%Y-%m-%d %H:%M:%S")}: {message.content}").tolist()

        #! DUMMY DATA
        instruct_embedding = [-5.1, 2.9, 0.8, 7.9, 3.1] # fruit
        embedding_vector = [-5.1, 2.9, 0.8, 7.9, 3.1] # fruit
        #! DUMMY DATA

        data = {
            # Transcriptions
            "trans_id": message.id, # snowflake id
            "timestamp": message.created_at,
            "timezone": "CT", #* change accordinly
            "speaker": str(message.author),
            "text": message.content,

            # Vectors
            "vec_id": sf.next_id(),
            "embedding_model": embedding_model,
            "embedding_dim": embedding_dim,
            "embedding": embedding_vector,
            "index_type": "StreamingDiskAnn", #* change accordingly
            "index_measurement": "vector_cosine_ops", #* change accordingly
        }

        # TODO: Call llm/langgraph for response and conditional querying
        #! prob need to change logic here
        err_message = "Error getting results"
        try:
            results = db.query_vector(query=instruct_embedding)
        except Exception as e:
            results = err_message
        
        try:
            await message.reply(f"These are the results: \n {results}", mention_author=True)
        except Exception as e:
            print(e)
            await message.reply(err_message, mention_author=True)

        try:
            db.add_record(table=TranscriptionsVectors,data=data)
            # save in all-data csv
            write_to_csv(full_file_path=all_records_csv_path, 
                        data=data)
        except Exception as e:
            print("Error: ", e)
            # save in not-added csv
            write_to_csv(full_file_path=not_added_csv_path, 
                        data=data)  
        #TODO

    
    # custom clean up when KeyboardInterrupted
    # https://stackoverflow.com/questions/69682471/how-do-i-gracefully-handle-ctrl-c-and-shutdown-discord-py-bot
    async def async_cleanup(self):
        #! Change table name as needed
        # create postgres -> csv copy
        tablename = "transcriptionsvectors"
        create_copy = input_to_bool(question="Create a backup csv file (export from database)? (y/n): ", true_options=true_options, false_options=false_options)

        if create_copy:
            # copy path
            #! Change path as needed
            today = datetime.now().strftime("%Y-%m-%d")
            copy_path = f'./db/backups/{tablename}_{today}'
            # create dir (input parents) if none exists
            Path(copy_path).mkdir(parents=True, exist_ok=True)

            # Validate file name
            copy_name = input("Copying data from PostgreSQL to csv. Give output csv name: ").lower().strip()
            while not is_valid_windows_name(copy_name):
                print(windows_filename_validity_message)
                copy_name = input("Copying data from PostgreSQL to csv. Give output csv name: ").lower().strip()

            # Compress file
            compress = input_to_bool(question="Compress file or not (y/n): ", true_options=true_options, false_options=false_options)

            # Export
            db.postgres_to_csv(table_name=tablename, output_path=copy_name, compress=compress)

        # Clean table (delete all rows)
        # Note: make sure you have a copy of the data before deleting
        clean = input_to_bool(question=f"Delete all rows from {tablename} table (y/n): ", true_options=true_options, false_options=false_options)
        if clean:
            clean_table(db=db, tablename=tablename, truncate=True)

        # Stop compose container
        yaml_path = "db/compose.yaml"
        close_docker_compose(compose_path=yaml_path, down=False)
    
    async def close(self):
        await self.async_cleanup()
        await super().close()  # don't forget this!


bot = MyBot(command_prefix="/", intents=discord.Intents.all())
bot.run(discord_token)