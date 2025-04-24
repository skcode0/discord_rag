import discord
from discord.ext import commands
from discord import app_commands
from dotenv import load_dotenv
import os

load_dotenv()

class Client(commands.Bot):
    async def on_ready(self):
        print(f"Logged on as {self.user}!")

        try:
            guild = discord.Object(1353418792444887100)
            synced = await self.tree.sync(guild=guild)
            print(f"Syned {len(synced)} commands to guild {guild.id}")
        except Exception as e:
            print(f"Error syncing commands: {e}")

    async def on_message(self, message):
        if message.author == self.user:
            return
        
        if message.content.startswith("hi"):
            time = message.created_at
            formatted_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            await message.channel.send(f"hi there, {message.author}. Created at {formatted_timestamp}. Message id: {message.id}")
    
    async def on_reaction_add(self, reaction, user):
        await reaction.message.channel.send('You reacted')

intents = discord.Intents.default()
intents.message_content = True

client = Client(command_prefix="!", intents=intents)

GUILD_ID = discord.Object(1353418792444887100)

@client.tree.command(name="hello", description="Says hello", guild=GUILD_ID)
async def sayHello(interaction: discord.Interaction):
    await interaction.response.send_message("Hi there")





client.run(os.environ.get("DISCORD_TOKEN"))