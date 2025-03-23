Discord messages are saved as vector embeddings and retrieved using StreamingDiskAnn.

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

<h1>Tools used:</h1>
- Docker
- discord.py
- PostgreSQL + pgAdmin 4
- SQLAlchemy + psycopg(3)
- PostgreSQL extensions: pgvector (vector) + pgvectorscale (StreamingDiskAnn)

