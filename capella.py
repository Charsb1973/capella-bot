import os
import discord

intents = discord.Intents.all()
client = discord.Client(intents=intents)
@client.event
async def on_ready():
  print(f"Le bot discord est prêt {client.user}")
@client.event
async def on_message(message):
  if message.author == client.user:
    return
  elif message.content.startswith('!bonjour'):
    await message.channel.send('Bonjour !')
@client.event
async def on_member_join(member):
  channel = client.get_partial_messageable(1366169551221293137)
  await channel.send(f"Bienvenue {member.name}")

bot = commands.Bot(command_prefix="!", intents=intents)

@bot.command()
async def ping(context):
  await context.send("Pong")
    
client.run(MTM2NjE3MTA4MDQwMTc1MjI0Nw.GAaW2g.i_lVkDKmMpixOQVU6-eRNOHo8M-DjlFDV09Dps)