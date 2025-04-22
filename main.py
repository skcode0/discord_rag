import os
from utils import (
    PostgresDataBase, 
    create_program_session_dir, 
    name_and_write_to_csv, 
    validate_ans, 
    write_to_csv, 
    clean_table, 
    close_docker_compose, 
    get_detailed_instruct, 
    create_embedding,
    update_file_num_pkl
    )
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
from tables import Base, Transcriptions, Vectors, TranscriptionsVectors

#* Note: This mostly uses synchronous functions. Async version of the code is in 'main_async.py'.

# --------------------------
# start Docker Compose command for DBs (only short term)
# --------------------------
command = ["docker", "compose", "db/compose.yaml", "up", "-d"]

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
program_session = create_program_session_dir() # create session folder

storage_path = './db/storage'

# For rows that were not successfully added to db for some reason (ex. duplicates, inserting null in non-nullable column, etc.)
not_added_file_name = "not_added.csv"
# for recording all data
all_file_name = "output.csv"

acceptable_ans = ["yes", "y", "no", "n"]
print("Following questions are for writing data (in csv) that has failed to save in database.")
date_input = validate_ans(acceptable_ans=acceptable_ans,
                        question="Add date? (y/n):")
if date_input in ["yes", "y"]:
    add_date = True
else:
    add_date = False

increment_input = validate_ans(acceptable_ans=acceptable_ans,
                       question="Add auto increment for file numbering? (y/n): ")
if increment_input in ["yes", "y"]:
    auto_increment = True
else:
    auto_increment = False

not_added_csv_path = name_and_write_to_csv(file_path=storage_path,
                                           file_name=not_added_file_name,
                                  session_name=program_session,
                                  add_date = add_date,
                                  auto_increment=auto_increment)

# This is for saving data in csv regardless of whether data was successfully added to database or not. In other words, this is simply a backup save.
print("----------")
print("Following questions are for writing all data to csv (back-up file).")
date_input = validate_ans(acceptable_ans=acceptable_ans,
                        question="Add date? (y/n):")
if date_input in ["yes", "y"]:
    add_date = True
else:
    add_date = False

increment_input = validate_ans(acceptable_ans=acceptable_ans,
                       question="Add auto increment for file numbering? (y/n): ")
if increment_input in ["yes", "y"]:
    auto_increment = True
else:
    auto_increment = False

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
                      db=short_db_name,
                      port=short_port)

url = db.make_db()
db.enable_vectors()


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

        # Create embedding
        # Note: Some embedding models like 'intfloat/multilingual-e5-large-instruct' require instructions to be added to query. Documents don't need instructions.
        task = "Given user's message query, retrieve relevant messages that answer the query."
        instruct_query = get_detailed_instruct(query=message.content,
                                                task_description=task)
        # for querying
        instruct_embedding = create_embedding(model_name=embedding_model,
                                              input=instruct_query)
        

        # for storage
        embedding_vector = create_embedding(model_name=embedding_model,
                                            input=f"{message.created_at.strftime("%Y-%m-%d %H:%M:%S")}: {message.content}").tolist()
        
        data = {
            # Transcriptions
            "timestamp": message.created_at,
            "speaker": message.author,
            "text": message.content,

            # Vectors
            "embedding_model": embedding_model,
            "embedding": embedding_vector,
            "index_type": "StreamingDiskAnn", #* change accordingly
            "index_measurement": "vector_cosine_ops", #* change accordingly
        }

        try:
            # TODO: Call llm/langgraph for response and conditional querying
            results = db.query_vector(query=instruct_embedding)
            await message.reply(f"These are the results: \n\n {results}", mention_author=True)

            db.add_record(table=TranscriptionsVectors,data=data)
            # save in all-data csv
            write_to_csv(full_file_path=all_records_csv_path, 
                        data=data)
        except Exception as e:
            print("Error: ", e)
            # save in not-added csv
            write_to_csv(full_file_path=not_added_csv_path, 
                        data=data)
            pass

    bot.run(discord_token)
    
except KeyboardInterrupt:
    # Clear short term memory data/rows
    # Decide if you want this table without checking. Recommend not deleting until all data is saved properly in csv or in other form(s).
    #! TODO test this
    tablename = "vectors"
    clean_table(db=db, tablename=tablename, truncate=True)

    # stop compose container
    yaml_path = "db/compose.yaml"
    # close_docker_compose(compose_path=yaml_path)
    
    raise

# *Note: Adding data to long_term_db will be done in 'long_term_db.py'.