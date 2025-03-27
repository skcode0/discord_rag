import os
from db.db_utils import make_pgdb
from sqlalchemy import create_engine, Index
from sqlalchemy.orm import DeclarativeBase, sessionmaker, mapped_column, Mapped
from pgvector.sqlalchemy import Vector
from datetime import datetime
import subprocess
import discord
from discord.ext import commands

# --------------------------
# start docker compose command for DB
# --------------------------
command = ["docker", "compose", "-f", "../db/compose.yaml", "up", "-d"]

try:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    print("Docker Compose Output:\n", result.stdout)
except subprocess.CalledProcessError as e:
    print("Error running docker-compose:", e.stderr)

# --------------------------
# Load in environment variables
# --------------------------
pg_username = os.environ.get('POSTGRESS_USER')
pg_password = os.environ.get('POSTGRES_PASSWORD')
db_name = os.environ.get('SHORT_TERM_DB')
port = os.environ.get('SHORT_TERM_HOST_PORT')
embedding_dim = int(os.environ.get('EMBEDDING_DIM'))
embedding_model = os.environ.get('EMBEDDING_MODEL')

# --------------------------
# Create database and table if not present
# --------------------------
# creates db if there's none
url = make_pgdb(password=pg_password,
                db=db_name,
                port=port,
                add_vectors=True)

engine = create_engine(url, echo=False)

class Base(DeclarativeBase):
    pass

# TODO
class Vectors(Base):
    # Reference: https://www.youtube.com/watch?v=iwENqqgxm-g&list=PLKm_OLZcymWhtiM-0oQE2ABrrbgsndsn0
    __tablename__ = "vectors"
    id: Mapped[int] = mapped_column(primary_key=True) # will auto increment
    time_spoken: Mapped[datetime]
    speaker: Mapped[str]
    text: Mapped[str]
    embedding = mapped_column(Vector(embedding_dim))

    # Reference: https://www.youtube.com/watch?v=WsDVBEmTlaI&list=PLKm_OLZcymWhtiM-0oQE2ABrrbgsndsn0&index=15
    # StreamingDiskAnn index
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


today = datetime.today().strftime('%Y_%m_%d')
# for rows that were not sucessfully added to db for some reason (ex. duplicates, inserting null in non-nullable column, etc.)
storage_path = '../db/storage'
not_added_path = storage_path + f"rows_not_added_{today}.csv" 
# for storing all data in csv format
path = storage_path + f"{today}"


# --------------------------
# Add discord messages as vector embeddings
# --------------------------

# TODO
# Session = sessionmaker(bind=engine)
# session = Session()


bot = commands.Bot(command_prefix="/", intents=discord.Intents.all())

@bot.event
async def on_ready():
    print(f"Logged on as {bot.user}!")

@bot.event
async def on_message(message):
    # ignore replaying to itself
    if message.author == bot.user:
        return
    
    # TODO
    await message.reply()



# bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())

# @bot.event
# async def on_ready():
#     print(f"Logged on as {bot.user}!")

# @bot.event
# async def on_message(message):
#     # ignore replying to itself
#     if message.author == bot.user:
#         return

#     await message.reply(inference_message(message.content), mention_author=True)


# bot.run(discord_bot_token)

