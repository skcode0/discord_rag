import os
from utils import make_pgdb, create_program_session_dir, create_postgres_url, name_and_write_to_csv, validate_ans
from sqlalchemy import create_engine, Index
from sqlalchemy.orm import DeclarativeBase, sessionmaker, mapped_column, Mapped
from pgvector.sqlalchemy import Vector
from datetime import datetime
import subprocess
import discord
from discord.ext import commands
from dotenv import load_dotenv
import sys

# --------------------------
# start Docker Compose command for DBs
# --------------------------
command = ["docker", "compose", "-f", "db/compose.yaml", "up", "-d"]

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
url = create_postgres_url(password=pg_password,
                          db_name=short_db_name,
                          port=short_port)

engine = create_engine(url, echo=False)

#! TODO: fix params
make_pgdb(engine=engine,
          password=pg_password,
          db=short_db_name,
          port=short_port,
          add_vectors=True)

class Base(DeclarativeBase):
    pass

# vectors table
class Vectors(Base):
    # Reference: https://www.youtube.com/watch?v=iwENqqgxm-g&list=PLKm_OLZcymWhtiM-0oQE2ABrrbgsndsn0
    __tablename__ = "vectors"
    id: Mapped[int] = mapped_column(primary_key=True) # will auto increment
    timestamp: Mapped[datetime]
    speaker: Mapped[str]
    text: Mapped[str]
    embedding = mapped_column(Vector(embedding_dim))

    # Reference: https://www.youtube.com/watch?v=WsDVBEmTlaI&list=PLKm_OLZcymWhtiM-0oQE2ABrrbgsndsn0&index=15
    # StreamingDiskAnn index (https://github.com/timescale/pgvectorscale/blob/main/README.md)
    __table_args__ = (
        Index(
            "embedding_idx",
            "embedding",
            postgresql_using="diskann",
            # postgresql_with={

            # } # index build parameters,
            postgresql_ops={"embedding": "vector_cosine_ops"} # cosine similarity
        ),
    )

Base.metadata.create_all(engine) # prevents duplicate tables

Session = sessionmaker(bind=engine)
# --------------------------
# Store Discord messages as embeddings (+ csv files) and call llm with rag to answer user inputs
# --------------------------

# TODO
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
        
        # TODO: do it in async
        # TODO: save message in csv (also create folder)
        # TODO: embed model
        # TODO: store messages as vectors in pg
        # TODO: call llm/langgraph for response and conditional querying
        await message.reply(f"{message.author} said: {message.content}", mention_author=True)

    bot.run(discord_token)
except KeyboardInterrupt:
    pass
    # TODO: process csv and vectors to long-term memory (create log)
    # TODO: stop compose container
    # TODO: clear short term memory data/rows
    # TODO: close discord bot
    # --------------------------
    # stop docker compose command for DB
    # --------------------------
