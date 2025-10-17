import discord
from PIL import Image
from discord.ext import commands
from discord import app_commands
import numpy as np
from numba import njit
import os
from enum import Enum

if not os.path.exists("temp"):
    os.makedirs("temp")

intents = discord.Intents.default()
intents.message_content = True
intents.dm_messages = True

bot = commands.Bot(command_prefix="!", intents=intents)


class FiltreOptions(Enum):
    floyd = "floyd"
    stucki = "stucki"
    ar = "ar"


@bot.event
async def on_ready():
    print(f"Connecté en tant que {bot.user}")
    try:
        synced = await bot.tree.sync()
        print(
            f"Commandes slash synchronisées : {[cmd.name for cmd in synced]}")
    except Exception as e:
        print(f"Erreur lors de la synchronisation des commandes : {e}")


@bot.command()
async def bonjours(ctx):
    await ctx.send("Salut tout le monde ! 👋")


@bot.tree.command(name="noir_et_blanc",
                  description="met ton image en noir et blanc !",
                  guild=None)
@app_commands.describe(
    error_type="choisir le type d'erreur diffusion (de base, prendre la 1er)",
    size=
    "met un multiplicateur de la taille de ton image (plus c'est grand, plus c'est quali), (pas plus de 5 sinon crash !)"
)
async def image_command(interaction: discord.Interaction,
                        error_type: FiltreOptions,
                        attachment: discord.Attachment,
                        size: float = 1.0):
    await interaction.response.defer(thinking=True)
    error_type = error_type.value
    if not attachment.filename.lower().endswith((".png", ".jpg", ".bmp")):
        await interaction.response.send_message(
            "l'image doit etre du format .png, .jpg ou .bmp")
        return
    if size > 5:
        await interaction.response.send_message(
            "size ne doit pas etre a plus de 5 pour ne pas faire tout crash")
        return
    path = f"temp/{attachment.filename}"
    await attachment.save(path)
    img = Image.open(path).convert('RGB')
    arr = np.array(img.resize(
        (int(img.width * float(size)),
         int(img.height * float(size)))) if size else img,
                   dtype=np.float32)
    arr = globals()[error_type](g(arr))
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr[:-2, 2:-2])
    img.save(path)
    await interaction.followup.send(f"Image traitée avec : {error_type}",
                                    file=discord.File(path))
    return


@bot.tree.command(name="ascii_art",
                  description="met ton image en ascii !",
                  guild=None)
@app_commands.describe(
    width="choisir la taille (max 100)",
    inverted="couleur inversé ?",
    quality=
    "met un multiplicateur de la taille de ton image (plus c'est grand, plus c'est quali), (pas plus de 5 sinon crash !)"
)
async def image_command(interaction: discord.Interaction,
                        width: int,
                        attachment: discord.Attachment,
                        quality: float = 1.0,
                        inverted: bool = False):
    await interaction.response.defer(thinking=True)
    inverted = not inverted
    if not attachment.filename.lower().endswith((".png", ".jpg", ".bmp")):
        await interaction.followup.send(
            "l'image doit etre du format .png, .jpg ou .bmp")
        return
    if quality > 5:
        await interaction.followup.send(
            "quality ne doit pas etre a plus de 5 pour ne pas faire tout crash"
        )
        return
    if width > 100:
        await interaction.response.send_message("hopopop ! pas plus de 100 !")
        return
    path = f"temp/{attachment.filename}"
    await attachment.save(path)
    img = Image.open(path).convert('RGB')
    arr = act(img, width, 1, quality, inverted)
    content = '```\n'
    content += '\n'.join(arr)
    content += '\n```'
    a = await safe_send_ascii(interaction.channel, content, interaction)
    if a:
        await interaction.followup.send(content)
    return


async def safe_send_ascii(channel, content, interaction):
    if not len(''.join(content)) + 8 + 2 * len(content) > 2000:
        return True
    content = content[5:-5]
    lines = content.split('\n')
    buffer = ""
    for line in lines:
        if len(buffer) + len(line) + 1 > 2000:
            await interaction.followup.send(
                buffer, allowed_mentions=discord.AllowedMentions.none())
            buffer = ""
        buffer += line + '\n'
    if buffer:
        await interaction.followup.send(
            buffer, allowed_mentions=discord.AllowedMentions.none())
    return False


@njit
def g(i):
    h, w = i.shape[:2]
    r = np.empty((h, w), np.float32)
    for y in range(h):
        for x in range(w):
            r[y,
              x] = .2989 * i[y, x][0] + .587 * i[y, x][1] + .114 * i[y, x][2]
    return r


@njit
def ar(a):
    h, w = a.shape
    for i in range(h - 2):
        for j in range(2, w - 2):
            v = a[i, j]
            n = 255. if v >= 128. else 0.
            e = v - n
            a[i, j] = n
            for x, y in [(0, 1), (1, -1), (1, 0), (1, 1), (2, 0), (0, 2)]:
                a[i + x, j + y] += e * .125
    return np.clip(a, 0, 255)


def floyd(a):
    h, w = a.shape
    for i in range(h - 1):
        for j in range(1, w - 1):
            v = a[i, j]
            n = 255. if v >= 128. else 0.
            e = v - n
            a[i, j] = n
            a[i, j + 1] += e * .4375
            a[i + 1, j - 1] += e * .1875
            a[i + 1, j] += e * .3125
            a[i + 1, j + 1] += e * .0625
    return np.clip(a, 0, 255)


def stucki(a):
    h, w = a.shape
    for i in range(h - 2):
        for j in range(2, w - 2):
            v = a[i, j]
            n = 255. if v >= 128. else 0.
            e = v - n
            a[i, j] = n
            for x, y in [(0, 1), (1, -1), (1, 0), (1, 1), (2, 0), (0, 2)]:
                a[i + x, j + y] += e * .125
    return np.clip(a, 0, 255)


def decouper_image(path_array, nb_colonnes):
    image = Image.fromarray(path_array[:-2, 2:-2], 'L')
    largeur, hauteur = image.size
    nb_colonnes = min(nb_colonnes, largeur)
    largeur_colonne = largeur // nb_colonnes
    hauteur_bloc = max(1, int(largeur_colonne * (5 / 3)))
    nb_blocs_verticaux = max(1, hauteur // hauteur_bloc)
    liste_blocs = []
    for col in range(nb_colonnes):
        x0, x1 = col * largeur_colonne, (col + 1) * largeur_colonne
        colonne_blocs = []
        for ligne in range(nb_blocs_verticaux):
            y0, y1 = ligne * hauteur_bloc, (ligne + 1) * hauteur_bloc
            bloc = image.crop((x0, y0, x1, y1))
            colonne_blocs.append(bloc)
        liste_blocs.append(colonne_blocs)
    return liste_blocs


def reduire_en_nb_2x4(image):
    image = image.convert("L")
    image = image.resize((2, 4), resample=Image.BICUBIC)
    return image.convert("1")


def image_vers_caractere_braille(image):
    pixels = image.load()
    mapping = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (0, 3), (1, 3)]
    code_braille = 0
    for i, (x, y) in enumerate(mapping):
        if pixels[x, y] == 0:
            code_braille |= (1 << i)
    return chr(0x2800 + code_braille)


def inverser_braille(car):
    code = ord(car)
    bits = code - 0x2800
    bits_inverse = bits ^ 0b11111111
    return chr(0x2800 + bits_inverse)


def full_invert(liste):
    return [[inverser_braille(car) for car in ligne] for ligne in liste]


def act(img, column, ver=1, quality=None, invert=False):
    arr = g(
        np.array(img.resize(
            (int(img.width * float(quality)),
             int(img.height * float(quality)))) if quality else img,
                 dtype=np.float32))
    if ver == 1:
        arr = floyd(arr)
    elif ver == 2:
        arr = ar(arr)
    else:
        arr = stucki(arr)
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    img = decouper_image(arr, column)
    img = [list(row) for row in zip(*img)]
    for i in range(len(img)):
        for j in range(len(img[0])):
            img[i][j] = image_vers_caractere_braille(
                reduire_en_nb_2x4(img[i][j]))
    if invert:
        img = full_invert(img)
    for i in range(len(img)):
        img[i] = ''.join(img[i])
    return img


from flask import Flask
from threading import Thread
import datetime

app = Flask('')


@app.route('/')
def home():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[PING] UptimeRobot ou navigateur a visité la page à {now}")
    return "Bot actif"


def run():
    app.run(host='0.0.0.0', port=8080)


def keep_alive():
    print('keepe alive')
    t = Thread(target=run)
    t.start()


keep_alive()
bot.run(os.getenv("token"))
