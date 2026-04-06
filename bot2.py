import os

import discord
from discord import app_commands
from dotenv import load_dotenv

from event import setup_events, setup_commands


load_dotenv()

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
if not DISCORD_BOT_TOKEN:
    raise ValueError("DISCORD_BOT_TOKEN이 .env에 없습니다.")

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.members = True

client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

setup_events(client, tree)
setup_commands(client, tree)

client.run(DISCORD_BOT_TOKEN)