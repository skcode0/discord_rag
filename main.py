import os
from utils import PostgresDataBase, create_program_session_dir, name_and_write_to_csv, validate_ans, write_to_csv, clean_table, close_docker_compose
from sqlalchemy import create_engine, Index
from sqlalchemy.orm import sessionmaker, mapped_column, Mapped
from pgvector.sqlalchemy import Vector
from datetime import datetime
import subprocess
import discord
from discord.ext import commands
from dotenv import load_dotenv
import sys
import asyncio
from tables import Base, Transcription, Vectors

#* Note: This mostly uses synchronous functions. Async version of the code is in 'main_async.py'.

# --------------------------
# start Docker Compose command for DBs (only short term)
# --------------------------
command = ["docker", "compose", "-f", "db/compose.yaml", "up", "-d" "short_term_db"]

try:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    print("Docker Compose Output:\n", result.stdout)
except subprocess.CalledProcessError as e:
    print("Error running docker-compose:", e.stderr)
    sys.exit(1)

# --------------------------
# Load in environment variables
# --------------------------
# postgres
load_dotenv()
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
create_program_session_dir() # create session folder

storage_path = './db/storage'

# for rows that were not sucessfully added to db for some reason (ex. duplicates, inserting null in non-nullable column, etc.)
not_added_file_name = "not_added.csv"
# for recording all data
all_file_name = "output.csv"

acceptable_ans = ["yes", "y", "no", "n"]
print("Following questions are for writing non-postgres-added data to csv.")
add_date = validate_ans(acceptable_ans=acceptable_ans,
                        question="Add date? Default is True. (y/n):")
auto_increment = input(acceptable_ans=acceptable_ans,
                       question="Add auto increment for file numbering? Default is False (y/n): ")
not_added_csv_path = name_and_write_to_csv(file_path=storage_path,
                                  session_name=program_session,
                                  add_date = True,
                                  auto_increment=False)

print("----------")
print("Following questions are for writing all data to csv.")
add_date = validate_ans(acceptable_ans=acceptable_ans,
                        question="Add date? Default is True. (y/n):")
auto_increment = input(acceptable_ans=acceptable_ans,
                       question="Add auto increment for file numbering? Default is False (y/n): ")
all_records_csv_path = name_and_write_to_csv(file_path=storage_path,
                                  session_name=program_session,
                                  add_date = True,
                                  auto_increment=False)

# --------------------------
# Create database (+ postgres extensions) and table if not present
# --------------------------
# short-term long-term db
db = PostgresDataBase(password=pg_password,
                      db=short_db_name,
                      port=short_port)

url = db.make_db()
db.enable_vectors()

# TODO
tablename = "intflat_multi_e5_large_inst"


# create table(s)
Base.metadata.create_all(db.engine) # prevents duplicate tables

Session = sessionmaker(bind=db.engine)

# --------------------------
# Store Discord messages as embeddings (+ csv files) and call llm with rag to answer user inputs
# --------------------------
try:
    bot = commands.Bot(command_prefix="/", intents=discord.Intents.all())

    @bot.event
    async def on_ready():
        print(f"Logged on as {bot.user}!")

    @bot.event
    async def on_message(message):
        # ignore replaying to itself
        if message.author == bot.user:
            return
        
        # TODO: call llm/langgraph for response and conditional querying
        await message.reply(f"{message.author} said: {message.content}", mention_author=True) #! Delete this later
        
        # add message as vector in postgres
        try:
            # TODO: create embedding
            embedding = 

            # TODO: store messages as vectors in pg
            data = {
                timestamp = message.created_at,
                speaker = message.author,
                text = message.content,
                embedding_model = embedding_model,
                embedding = embedding,
            }

            # vectors tables need to be dynamic
            db.add_record(table=Vectors, data=data)
            # save in all-data csv
            write_to_csv(full_file_path=all_records_csv_path, 
                        data=data)
        except Exception as e:
            # save in not-added csv
            write_to_csv(full_file_path=not_added_csv_path, 
                        data=data)
            pass

    bot.run(discord_token)
    
except KeyboardInterrupt:
    # clear short term memory data/rows
    tablename = "vectors"
    clean_table(db=db, tablename=tablename, truncate=True)

    # stop compose container
    yaml_path = "db/compose.yaml"
    close_docker_compose(compose_path=yaml_path)
    
    raise

# *Note: Adding data to long_term_db will be done in 'long_term_db.py'.