<h1>Overview</h1>

- A project to learn about RAG, vector DBs, ReAct agents, and LangGraph.

- Discord messages are saved as vector embeddings and retrieved using StreamingDiskAnn. This is tested on PostgreSQL only.


<h2>3 main components:</h2>

1. Discord + LLM/ReAct Agents
2. Short-term database: only saves messages from current program session. This database should refresh every new program session.
3. Long-term database: stores ALL messages. This database shouldn't change much.

- `main_async.py` runs the main program: Discord, short-term memory (for storage and querying), long-term memory (for quering) and LLM.
- `backups.py` is for backing up database. It will create .csv and .dump files. Recommended for short-term memory db where it will constantly need to refresh memories for each program session.
- `db_save` is for loading up csv file into pandas, cleaning/processing data, and then sending that data to PostgreSQL. Recommended for sending short-term data to long-term db.


<h1> Other Notes </h1>

- **IMPORTANT**: Create `.env` file. Use `.env_template` for reference.

- **IMPORTANT**: Create `storage` folder inside `db` folder.

- `sample.csv` is example data you can use for testing out vector databases.

- Before running Discord, it will ask for a program session name and if you want backup csv files to add date and put incremental number at the end if there's duplicate. You can keep on using the same program session or rename it. If the rename is empty, then it will create a randomly generated name. The program session folders are created inside `db/storage` directory. If there's `db/storage`, create one. Inside each program session, you will have 3 files (you can rename them to whatever): csv for storing all messages, csv for storing messages that short-term database failed to add for some reason, and pickle file for incremental file numbering. For the pickle file, if you delete some csv files and/or pickle file, you can rebuild the incremental file numbering using `update_file_num_pkl` function in `util_async.py`.

- **IMPORTANT**: Although the functions dealing with files/folders work, they can be buggy. They are NOT designed to catch all cases. For instance, the file/folder names should adhere to `{valid_file_name}_{%Y-%m-%d}_{file_number}` or you can run into errors or get unexpected results. Also, these functions aren't really designed to be the most efficient and optimal. They work (kinda..), and it shouldn't be much problem for not-so-big file/folder structures.

- In Discord, every message, including slash command texts, will be saved to the database AND inside csv file. To call the LLM, you must type `/chat` and then write whatever you want. It will use tools if necessary to call up databases and respond. The program will run until you keyboard interrupt (ctrl+c).

- I decided not to backup databases and add data to long term db inside main for 2 reasons:
    1. Discord can timeout for bot responses that take too long. Although I don't think it's much of a problem, I just didn't want to deal with it.
    2. You are adding/deleting/modifying databases, so it's probably better to have more control inside separate .py files.

- Look at log files to see history.

- The program essentially creates 3 backups:
    1. csv while running main program
    2. csv backup from `backups.py`
    3. `.dump` from `backups.py`
    You can disable any of them to save space.

- You may have noticed that some functions may be running asynchronously unnecessarily, or that they are not truly async -- as in, they are running on thread (ex. aiofile vs aiofiles). Yes, this adds overhead and can make the code run slower, but there's so many API calls and file I/O that I made the functions (NOT ALL) run in async.


<h1> Discord Bot </h1>

<h2> Create Discord Bot: </h2>

- Create a new server for the bot if you want.
- Go to `Discord Developer Portal` on web browser.
- Log in with discord account.
- Click on `New Application` button.
- Give it a name and create.
- Go to `Bot` tab to create a bot.
- Change name, profile pic, banner of the bot if you want.
- IMPORTANT: scroll down and enable ALL intents (presence, server members, message content). This allows bot to interact with server.
- IMPORTANT: Go to `Bot Permissions` and check `Administrator`.
- Save changes.

<h2> Invite Bot to Discord server: </h2>

- Go to `OAuth2` tab.
- Scroll to `OAuth2 URL Generator` and click `bot`.
- Scroll to `Bot Permissions` and click `Administrator`
- You'll see `generated url` with a unique link. Copy and paste the link on web browser. It'll redirect to discord app and ask for permissions (make sure to choose the right server for the bot).
- After given permission, Bot should be in the server (offline)
- After you put the intial code, you'll need to put in token to actually run the bot. You can find that token by going to the `Bot` tab again, clicking on `Reset Token`, do authentication, and `Copy` token. Paste it inside `.env` file and name it `DISCORD_TOKEN`. 

<h2>Dev Mode and Server ID</h2>

- Go to your Discord account settings and go to `Advanced` tab. Enable `Developer Mode`.
- When using slash commands, it's recommended to only load commands for a specific server, at least for developmental phase anyway. So right click on server where the bot resides in and click on `Copy Server ID`. This will give you server/guild id that you can use it for slash commands. 


<h1>Tools used:</h1>
Note: look at `requirements.txt` to see all the installed packages.

- Docker
- discord.py
- PostgreSQL + pgAdmin 4
- SQLAlchemy + psycopg(3)
- PostgreSQL extensions: pgvector (vector) + pgvectorscale (StreamingDiskAnn)
- SonyFlake
- Pandas/Numpy
- Hugging Face
- PyTorch
- Langgraph
- asyncio


