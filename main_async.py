import os
from utils_async import (
    AsyncPostgresDataBase, 
    create_program_session_dir, 
    name_and_write_to_csv,
    write_to_csv, 
    close_docker_compose, 
    get_detailed_instruct, 
    create_embedding,
    input_to_bool,
    )
from datetime import datetime, timezone
import subprocess
import discord
from discord.ext import commands
from dotenv import load_dotenv
import sys
from tables import CombinedBase, TranscriptionsVectors
from sonyflake import SonyFlake
import asyncio
import textwrap


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
discord_server_id = int(os.environ.get('DISCORD_SERVER_ID'))

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

async def main():
    # --------------------------
    # Create database (+ postgres extensions) and table if not present
    # --------------------------
    #TODO: open up long_term db (for querying only)
    # long-term db
    long_db = AsyncPostgresDataBase(password=pg_password,
                        user=pg_username,
                        db_name=long_db_name,
                        port=long_port,
                        hide_parameters=True)

    # short-term db
    short_db = AsyncPostgresDataBase(password=pg_password,
                        user=pg_username,
                        db_name=short_db_name,
                        port=short_port,
                        hide_parameters=True)

    await short_db.make_db()
    await short_db.enable_vectors()

    # create table(s)
    async with short_db.engine.begin() as conn:
        await conn.run_sync(CombinedBase.metadata.create_all) # prevents duplicate tables

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
            
            await short_db.add_record(table=TranscriptionsVectors, data=data)
    except:
        pass
    #! DUMMY DATA

    # --------------------------
    # Store Discord messages as embeddings (+ csv files) and call llm with rag to answer user inputs
    # --------------------------
    GUILD_ID = discord.Object(id=discord_server_id)
    class MyBot(commands.Bot):
        async def on_ready(self):
            print(f"Logged on as {bot.user}!")

            try:
                synced = await self.tree.sync(guild=GUILD_ID)
                print(f"Syned {len(synced)} commands to guild {GUILD_ID.id}.")
            except Exception as e:
                print(f"Error syncing commands: {e}")
                raise

        async def on_message(self, message):
            # Note: If only images/videos/audio/gifs or any other attachments sent, text content will be empty.
            # Note: If storing unstructured data, use multi-modal embedding model (and data lake).
            if message.content != "":
                # for storage
                # embedding_vector = await create_embedding(model_name=embedding_model,
                #                                     input=f"{message.created_at.strftime("%Y-%m-%d %H:%M:%S")}: {message.content}").tolist()

                #! DUMMY DATA
                embedding_vector = [-5.1, 2.9, 0.8, 7.9, 3.1] # fruit
                #! DUMMY DATA

                data = {
                    # Transcriptions
                    "trans_id": message.id, # snowflake id
                    "timestamp": message.created_at,
                    "timezone": "CT", #* change accordingly
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
                
                try:
                    async with asyncio.TaskGroup() as tg:
                        # add to db
                        tg.create_task(short_db.add_record(table=TranscriptionsVectors,data=data))
                        # save in all-data csv
                        tg.create_task(write_to_csv(full_file_path=all_records_csv_path, 
                                    data=data))     
                except* Exception as eg:
                    for e in eg.exceptions:
                        print("Error:", e)
                    # save in not-added csv
                    await write_to_csv(full_file_path=not_added_csv_path, 
                                data=data)

        
        # custom clean up when KeyboardInterrupted
        # https://stackoverflow.com/questions/69682471/how-do-i-gracefully-handle-ctrl-c-and-shutdown-discord-py-bot
        async def async_cleanup(self):
            # Stop compose container
            yaml_path = "db/compose.yaml"
            await close_docker_compose(compose_path=yaml_path, down=False)
        
        async def close(self):
            await self.async_cleanup()
            await super().close()  # don't forget this!


    bot = MyBot(command_prefix="/", intents=discord.Intents.all())


    # Slash Commands and Deferred Replies
    # https://www.youtube.com/watch?v=JN5ya4mMkek
    GUILD_ID = discord.Object(id=discord_server_id)
    @bot.tree.command(name="chat", description="Chat with AI bot.", guild=GUILD_ID)
    async def chat(interaction: discord.Interaction, text: str):
        await interaction.response.defer()
        
        # TODO: Call llm/langgraph for response and conditional querying

        # Create embedding
        # Note: Some embedding models like 'intfloat/multilingual-e5-large-instruct' require instructions to be added to query. Documents don't need instructions.
        #! Delete/edit query embedding instruction as needed.
        task = "Given user's message query, retrieve relevant messages that answer the query."
        # instruct_query = get_detailed_instruct(query=message.content,
        #                                         task_description=task)
        # # for querying
        # instruct_embedding = create_embedding(model_name=embedding_model,
        #                                       input=instruct_query)

        #! DUMMY DATA
        instruct_embedding = [-5.1, 2.9, 0.8, 7.9, 3.1] # fruit
        #! DUMMY DATA

        err_message = "Error getting results"
        try:
            #TODO: will change this logic later
            short_result = await short_db.query_vector(query=instruct_embedding)
            long_results = await long_db.query_vector(query=instruct_embedding)
        except Exception as e:
            results = err_message
        
        response = "Some llm response"
        #TODO----
        response = results

        limit = 2000 # message char limit
        if len(response) > limit:
            tw = textwrap.TextWrapper(
                width=limit,
                break_long_words=True,
                break_on_hyphens=True
            )
            
            chunks = tw.wrap(response)
            chunks_len = len(chunks)
            for i,c in enumerate(chunks, 1):
                await interaction.followup.send(f"Page {i}/{chunks_len}: {interaction.user.mention} {c}")
        else:
            await interaction.followup.send(f"{interaction.user.mention} {response}")
        

    bot.run(discord_token)

asyncio.run(main())

#* Postgres -> csv and db backups: run 'backups.py'
#* csv -> Postgres: run 'db_save.py'