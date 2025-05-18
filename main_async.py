import os
from utils_async import (
    AsyncPostgresDataBaseUser,
    AsyncPostgresDataBaseSuperUser, 
    create_program_session_dir, 
    name_and_write_to_csv,
    write_to_csv_async, 
    close_docker_compose, 
    get_detailed_instruct, 
    create_embedding,
    input_to_bool,
    create_hypertable_ddl
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
from langgraph_workflow import app
from langchain_core.messages import HumanMessage, SystemMessage

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
# bot access
bot_user = os.environ.get('POSTGRESS_BOT_USER')
bot_password = os.environ.get('POSTGRES_BOT_PASSWORD')

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

#! Change names as needed
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
# long-term db (only to give access to bot)
long_db = AsyncPostgresDataBaseSuperUser(password=pg_password,
                    user=pg_username,
                    db_name=long_db_name,
                    port=long_port,
                    hide_parameters=True)


# short-term db
short_db = AsyncPostgresDataBaseSuperUser(password=pg_password,
                    user=pg_username,
                    db_name=short_db_name,
                    port=short_port,
                    hide_parameters=True)

short_db.make_db()

# bot
bot_short = AsyncPostgresDataBaseUser(password=bot_password,
                    user=bot_user,
                    db_name=short_db_name,
                    port=short_port,
                    hide_parameters=True)
bot_long = AsyncPostgresDataBaseUser(password=bot_password,
                    user=bot_user,
                    db_name=long_db_name,
                    port=short_port,
                    hide_parameters=True)

# pk generation 
#! Change start_date, machine_id if necessary
start_time = datetime(2025,1,1,tzinfo=timezone.utc)
sf = SonyFlake(start_time=start_time, machine_id=lambda: 1)


# --------------------------
# Store Discord messages as embeddings (+ csv files) and call llm with rag to answer user inputs
# --------------------------
GUILD_ID = discord.Object(id=discord_server_id)
class MyBot(commands.Bot):
    async def on_ready(self):
        print(f"Logged on as {bot.user}!")

        try:
            synced = await self.tree.sync(guild=GUILD_ID)
            print(f"Synced {len(synced)} commands to guild {GUILD_ID.id}.")
        except Exception as e:
            print(f"Error syncing commands: {e}")
            raise

    async def on_message(self, message):
        # Note: If only images/videos/audio/gifs or any other attachments sent, text content will be empty.
        # Note: If storing unstructured data, use multi-modal embedding model (and data lake).
        if message.content != "":
            # for storage
            embedding_vector = await create_embedding(model_name=embedding_model, input=message.content)
            embedding_vector = await asyncio.to_thread(embedding_vector.tolist)

            #! DUMMY DATA
            # embedding_vector = [-5.1, 2.9, 0.8, 7.9, 3.1] # fruit
            #! DUMMY DATA

            data = {
                # Transcriptions
                "trans_id": message.id, # snowflake id
                "timestamp": message.created_at, # UTC
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
                    tg.create_task(write_to_csv_async(full_file_path=all_records_csv_path, 
                                data=data))
            except* Exception as eg:
                for e in eg.exceptions:
                    print("Error:", e)
                # save in not-added csv
                await write_to_csv_async(full_file_path=not_added_csv_path, 
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

@bot.tree.command(name="about", description="Info about the bot.", guild=GUILD_ID)
async def chat(interaction: discord.Interaction):
    await interaction.response.send_message("""
    This bot can access databases to retrieve any relevant discord messages and respond accordingly. Note that the bot will ONLY save and know about the messages starting from it's first run and while running. So any messages sent before inviting and while not running the bot won't be considered.                     
    """)


system_prompt = f"""You are a helpful assistant that can use tools to respond to user. Use however many tools needed to respond to user's input. Make sure to use markdown format for Discord.
"""

# Note: Some embedding models like 'intfloat/multilingual-e5-large-instruct' require instructions to be added to query. Documents don't need instructions.
#! Deal with query embedding instruction as needed.
task = "Given user's message query, retrieve relevant messages that answer the query."

@bot.tree.command(name="chat", description="Chat with AI bot.", guild=GUILD_ID)
async def chat(interaction: discord.Interaction, text: str):
    await interaction.response.defer()

    #TODO---------------------- 
    # vector embedding

    instruct_query = get_detailed_instruct(query=text,
                                            task_description=task)
    
    async with asyncio.TaskGroup() as tg:
        # embedding for db
        task_emb = tg.create_task(create_embedding(model_name=embedding_model, input=text))
        # query embed
        task_inst = tg.create_task(create_embedding(model_name=embedding_model, input=instruct_query))
    
    embedding_vector = task_emb.result()
    instruct_embedding = task_inst.result()

    # convert to list
    embedding_vector, instruct_embedding = await asyncio.gather(
    asyncio.to_thread(embedding_vector.tolist), asyncio.to_thread(instruct_embedding.tolist))

    
    # #! DUMMY DATA
    # embedding_vector = np.random.rand(1024).tolist()
    # instruct_embedding = np.random.rand(5).tolist()
    # embedding_vector = [-0.1, 4.3, 45.8, -37.94, 1.1]
    # #! DUMMY DATA

    data = {
        # Transcriptions
        "trans_id": interaction.id, # snowflake id
        "timestamp": interaction.created_at, # UTC
        "speaker": str(interaction.user),
        "text": text,

        # Vectors
        "vec_id": sf.next_id(),
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "embedding": embedding_vector,
        "index_type": "StreamingDiskAnn", #* change accordingly
        "index_measurement": "vector_cosine_ops", #* change accordingly
    }


    initial_state = {
        "embedding": instruct_embedding,
        "messages": [SystemMessage(content=system_prompt),
                    HumanMessage(content=text)]
    }

    # for in-memory
    config = {"configurable": {"thread_id": program_session}}

    try:
        async with asyncio.TaskGroup() as tg:
            #TODO: test this
            agent_task = tg.create_task(app.ainvoke(initial_state, config))
            # add to db
            tg.create_task(short_db.add_record(table=TranscriptionsVectors,data=data))
            # save in all-data csv
            tg.create_task(write_to_csv_async(full_file_path=all_records_csv_path, 
                        data=data))
    except* Exception as eg:
        for e in eg.exceptions:
            print("Error:", e)
        # save in not-added csv
        await write_to_csv_async(full_file_path=not_added_csv_path, 
                    data=data)

    response = agent_task.result()
    response = response["messages"][-1].content
    print(response)
    print(len(response))
    
    #TODO----------------------
    limit = 1900 # message char limit
    if len(response) > limit:
        tw = textwrap.TextWrapper(
            width=limit,
            break_long_words=True,
            break_on_hyphens=True
        )
        
        chunks = tw.wrap(response)
        chunks_len = len(chunks)
        for i,c in enumerate(chunks, 1):
            try:
                await interaction.followup.send(f"Page {i}/{chunks_len}: {interaction.user.mention} {c}")
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"[ERROR] Failed to send chunk {i}: {e}")
    else:
        await interaction.followup.send(f"{interaction.user.mention} {response}")

# create hypertable (time-based partitioning)
#! Change chunk time interval if needed
create_hypertable_ddl(table=TranscriptionsVectors, time_col="timestamp", chunk_interval="1 sec")

# --------------------------
# Run main()
# --------------------------
async def main():
    await short_db.enable_vectors()

    # bot access for long-term
    group_name = "readonly"
    await long_db.create_readonly_group(db_name=long_db_name, group_name=group_name)
    await long_db.add_user(role_name=bot_user, group_name=group_name, password=bot_password)
    # bot access for short-term
    await short_db.create_readonly_group(db_name=short_db_name, group_name=group_name)
    await short_db.add_user(role_name=bot_user, group_name=group_name, password=bot_password)


    # create table(s)
    async with short_db.engine.begin() as conn:
        await conn.run_sync(CombinedBase.metadata.create_all) # prevents duplicate tables

    #! DUMMY DATA
    # # https://weaviate.io/blog/vector-embeddings-explained
    # emb = [
    #         ["cat", [1.5, -0.4, 7.2, 19.6, 20.2]],
    #         ["dog", [1.7, -0.3, 6.9, 19.1, 21.1]],
    #         ["apple", [-5.2, 3.1, 0.2, 8.1, 3.5]],
    #         ["strawberry", [-4.9, 3.6, 0.9, 7.8, 3.6]],
    #         ["building",[60.1, -60.3, 10, -12.3, 9.2]],
    #         ["car",[81.6, -72.1, 16, -20.2, 102]]
    # ]

    # snowflake_id = [7320991239237537792, 7320991239237537793, 7320991239237537794, 7320991239237537795, 7320991239237537796, 7320991239237537797]

    # try:
    #     for i in range(len(emb)):
    #         data = {
    #                     # Transcriptions
    #                     "trans_id": snowflake_id[i],
    #                     "timestamp": datetime.now(timezone.utc),
    #                     "speaker": f"user_{i}",
    #                     "text": emb[i][0],

    #                     # Vectors
    #                     "vec_id": sf.next_id(),
    #                     "embedding_model": embedding_model,
    #                     "embedding_dim": embedding_dim,
    #                     "embedding": emb[i][1],
    #                     "index_type": "StreamingDiskAnn", #* change accordingly
    #                     "index_measurement": "vector_cosine_ops", #* change accordingly
    #                 }

    #         await short_db.add_record(table=TranscriptionsVectors, data=data)
    # except Exception as e:
    #     # raise e
    #     pass
    # #! DUMMY DATA

    try:
        await bot.start(discord_token)
    finally:
        await bot.close()
        sys.exit(0)


asyncio.run(main())


#* Postgres -> csv and db backups: run 'backups.py'
#* csv -> Postgres: run 'db_save.py'