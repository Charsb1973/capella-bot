#-----------------Import

import discord
from PIL import Image
from discord.ext import commands
from discord import app_commands
import numpy as np
from numba import njit
import os
from enum import Enum
import cv2
import svgwrite
import random

#--------------------Discord_Bot_Setup-----

if not os.path.exists("temp"):
    os.makedirs("temp")

intents = discord.Intents.default()
intents.message_content = True
intents.dm_messages = True

bot = commands.Bot(command_prefix="!", intents=intents)

#--------------------When_ready----------

@bot.event
async def on_ready():
    print(f"Connecté en tant que {bot.user}")
    try:
        synced = await bot.tree.sync()
        print(f"Commandes slash synchronisées : {[cmd.name for cmd in synced]}")
    except Exception as e:
        print(f"Erreur lors de la synchronisation des commandes : {e}")

#---------------Black_white_err_diff-------
    
class FiltreOptions(Enum):
    floyd = "floyd"
    stucki = "stucki"
    ar = "ar"

@bot.tree.command(name="black_and_white", description="put your img in black and white !", guild=None)
@app_commands.describe(error_type="choose error diffusion type !",size="bigger = best quality, (no more than 5 !)")
async def image_command(interaction: discord.Interaction,error_type: FiltreOptions,attachment: discord.Attachment, size: float=1.0):
    await interaction.response.defer(thinking=True)
    error_type=error_type.value
    if not attachment.filename.lower().endswith((".png", ".jpg", ".bmp")):
        await interaction.followup.send("img need to be .png, .jpg ou .bmp")
        return
    if size>5:
        await interaction.followup.send("no more than 5 !")
        return
    path=f"temp/{attachment.filename}"
    await attachment.save(path)
    img=Image.open(path).convert('RGB')
    arr=np.array(img.resize((int(img.width*float(size)),int(img.height*float(size)))) if size else img,dtype=np.float32)
    arr=globals()[error_type](g(arr))
    arr=(arr*255).clip(0,255).astype(np.uint8)
    img=Image.fromarray(arr[:-2, 2:-2])
    img.save(path)
    await interaction.followup.send(f"Image traitée avec : {error_type}", file=discord.File(path))
    return

#-------------Ascii_Art----------------
#|-> transform images in binary, with a selected error diffusion system
#>>> cut images in a lot of small pictures, each picture for an ascii character
#>>> apply an other error_diffusion on the small picture to get a 8 by 2 pixel img
#>>> find the correct braille character that fit with the picture
#>>> send all braille characters to recreate the image.
@bot.tree.command(name="ascii_art", description="met ton image en ascii !", guild=None)
@app_commands.describe(width="choisir la taille (max 100)",inverted="couleur inversé ?",quality="met un multiplicateur de la taille de ton image (plus c'est grand, plus c'est quali), (pas plus de 5 sinon crash !)")
async def image_command(interaction: discord.Interaction,width: int,attachment: discord.Attachment, quality: float=1.0, inverted: bool=False):
    await interaction.response.defer(thinking=True)
    inverted=not inverted
    if not attachment.filename.lower().endswith((".png", ".jpg", ".bmp")):
        await interaction.followup.send("l'image doit etre du format .png, .jpg ou .bmp")
        return
    if quality>5:
        await interaction.followup.send("quality ne doit pas etre a plus de 5 pour ne pas faire tout crash")
        return
    if width>100:
        await interaction.response.send_message("hopopop ! pas plus de 100 !")
        return
    path=f"temp/{attachment.filename}"
    await attachment.save(path)
    img=Image.open(path).convert('RGB')
    arr=act(img, width, 1, quality, inverted)
    content='```\n'
    content += '\n'.join(arr)
    content += '\n```'
    a= await safe_send_ascii(interaction.channel, content, interaction)
    if a:
        await interaction.followup.send(content)
    return

#->>> ----safe_send----
#|-> Permite to send messages with more than 2000 characters
#     >>> if lenght is > 2000, we send others messages, to divide the text in different message with less than 2000 characters
async def safe_send_ascii(channel, content, interaction):
    if not len(''.join(content))+8+2*len(content)>2000:
        return True
    content=content[5:-5]
    lines = content.split('\n')
    buffer = ""
    for line in lines:
        if len(buffer) + len(line) + 1 > 2000:
            await interaction.followup.send(buffer, allowed_mentions=discord.AllowedMentions.none())
            buffer = ""
        buffer += line + '\n'
    if buffer:
        await interaction.followup.send(buffer, allowed_mentions=discord.AllowedMentions.none())
    return False

#-----------Vectorize----------------

@bot.tree.command(name="vectorize", description="met ton image en vectorielle !", guild=None)
async def image_command(interaction: discord.Interaction,attachment: discord.Attachment):
    await interaction.response.defer(thinking=True)
    if not attachment.filename.lower().endswith((".png", ".jpg", ".bmp")):
        await interaction.response.send_message("l'image doit etre du format .png, .jpg ou .bmp")
        return
    path=f"temp/{attachment.filename}"
    await attachment.save(path)
    vector(path, "temp")
    svg_to_png('temp/svg_couleur.svg','temp/apercu.png')
    await interaction.followup.send("Aperçu vectoriel :", files=[discord.File('temp/apercu.png'),discord.File('temp/svg_couleur.svg')])
    return

#---------seuillage_black_white-----

@bot.tree.command(name="noir_blanc_seuillage", description="met ton image en noir et blanc par niveau de seuil !", guild=None)
@app_commands.describe(seuil="choisir le seuil (optionel, de 0 à 255)")
async def image_command(interaction: discord.Interaction,attachment: discord.Attachment,seuil: int=128):
    await interaction.response.defer(thinking=True)
    if not attachment.filename.lower().endswith((".png", ".jpg", ".bmp")):
        await interaction.response.send_message("l'image doit etre du format .png, .jpg ou .bmp")
        return
    path=f"temp/{attachment.filename}"
    await attachment.save(path)
    img=seuillage_compact(Image.open(path),seuil)
    img.save(path)
    await interaction.followup.send(f"noir et blanc de seuil {seuil} :", file=discord.File(path))
    return

#----------choose-------------------

@bot.tree.command(name="choose", description="choisis une option aléatoire entre deux valeur", guild=None)
@app_commands.describe(choix_1='choisir 1er option',choix_2='choisir 2eme option',choix_3='choisir 3eme option',choix_4='choisir 4eme option',choix_5='choisir 5eme option',choix_6='choisir 6eme option',choix_7='choisir 7eme option',choix_8='choisir 8eme option',choix_9='choisir 9eme option')    
async def image_command(interaction: discord.Interaction,choix_1: str,choix_2: str,choix_3: str=None,choix_4: str=None,choix_5: str=None,choix_6: str=None,choix_7: str=None,choix_8: str=None,choix_9: str=None):
    a=[]
    b=[choix_1,choix_2,choix_3,choix_4,choix_5,choix_6,choix_7,choix_8,choix_9]
    a += [i for i in b if i is not None]
    await interaction.response.send_message(f"je choisis {random.choice(a)} !")
    
#----------roll---------------------
    
@bot.tree.command(name="roll", description="roule un dée au nombre de face que tu veux !", guild=None)
@app_commands.describe(nombre='choisis le nombre de face du dée !')    
async def image_command(interaction: discord.Interaction,nombre:int=6):
    await interaction.response.send_message(f'le dée est tombé sur {random.randint(1,nombre)} !')
    


@njit
def color_to_grey(image):
    height, width = image.shape[:2]
    new_array = np.empty((height, width), np.float32)
    for y in range(height):
        for x in range(width):
            new_array[y,
              x] = .2989 * image[y, x][0] + .587 * image[y, x][1] + .114 * image[y, x][2]
    return new_array


@njit
def ar(array):
    height, width = array.shape
    for i in range(height - 2):
        for j in range(2, width - 2):
            pixel = array[i, j]
            new_pixel = 255. if pixel >= 128. else 0.
            error = pixel - new_pixel
            array[i, j] = new_pixel
            for x, y in [(0, 1), (1, -1), (1, 0), (1, 1), (2, 0), (0, 2)]:
                array[i + x, j + y] += error * .125
    return np.clip(array, 0, 255)


def floyd(array):
    height, width = array.shape
    for i in range(height - 1):
        for j in range(1, width - 1):
            pixel = array[i, j]
            new_pixel = 255. if pixel >= 128. else 0.
            error = pixel - new_pixel
            array[i, j] = new_pixel
            array[i, j + 1] += error * .4375
            array[i + 1, j - 1] += error * .1875
            array[i + 1, j] += error * .3125
            array[i + 1, j + 1] += error * .0625
    return np.clip(array, 0, 255)


def stucki(array):
    height, width = array.shape
    for i in range(height - 2):
        for j in range(2, width - 2):
            pixel = array[i, j]
            new_pixel = 255. if pixel >= 128. else 0.
            error = pixel - new_pixel
            array[i, j] = new_pixel
            for x, y in [(0, 1), (1, -1), (1, 0), (1, 1), (2, 0), (0, 2)]:
                array[i + x, j + y] += error * .125
    return np.clip(array, 0, 255)

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
    mapping = [(0, 0), (0, 1), (0, 2), (1, 0),
               (1, 1), (1, 2), (0, 3), (1, 3)]
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
    arr=color_to_grey(np.array(img.resize((int(img.width*float(quality)),int(img.height*float(quality)))) if quality else img,dtype=np.float32))
    if ver == 1:
        arr = floyd(arr)
    elif ver == 2:
        arr = ar(arr)
    else:
        arr = stucki(arr)
    arr=(arr*255).clip(0,255).astype(np.uint8)
    img = decouper_image(arr, column)
    img = [[img[i][j] for i in range(len(img))] for j in range(len(img[0]))]
    for i in range(len(img)):
        for j in range(len(img[0])):
            img[i][j] = image_vers_caractere_braille(reduire_en_nb_2x4(img[i][j]))
    if invert:
        img = full_invert(img)
    for i in range(len(img)):
        img[i] = ''.join(img[i]) 
    return img

def vector(path,s_path):
    img_color = cv2.imread(path)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray, 50, 150)
    kernel = np.ones((2, 2), np.uint8)
    edges_thick = cv2.dilate(edges, kernel, iterations=2)
    edges_inv = cv2.bitwise_not(edges_thick)
    contours, _ = cv2.findContours(edges_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = img_gray.shape
    svg_full_path = os.path.join(s_path, "svg_couleur.svg")
    dwg = svgwrite.Drawing(svg_full_path, size=(f"{w}px", f"{h}px"))
    dwg.add(dwg.rect(insert=(0, 0), size=(f"{w}px", f"{h}px"), fill="black"))
    for contour in contours:
        if cv2.contourArea(contour) < 10:  # ignorer les toutes petites zones
            continue
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)  # remplir la forme
        mean_color = cv2.mean(img_color, mask=mask)[:3]  # BGR
        hex_color = '#%02x%02x%02x' % (int(mean_color[2]), int(mean_color[1]), int(mean_color[0]))
        points = [(int(p[0][0]), int(p[0][1])) for p in contour]
        if points[0] != points[-1]:
            points.append(points[0])
        dwg.add(dwg.polygon(points, fill=hex_color, stroke='black', stroke_width=0.5))
    dwg.save(s_path)    
    
def extraire_valeur(champ,ligne):
    debut = ligne.find(f'{champ}="') + len(champ) + 2
    fin = ligne.find('"', debut)
    return ligne[debut:fin]
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def svg_to_png(path,s_path):
    with open(path, "r", encoding="utf-8") as f:
        contenu = f.read()
    contenu = contenu.splitlines()[3:]
    back_color = extraire_valeur('fill', contenu[0])
    height = int(extraire_valeur('height', contenu[0])[:-2])
    width = int(extraire_valeur('width', contenu[0])[:-2])
    image = np.ones((height*2,width*2, 3), dtype=np.uint8)*0
    poly=contenu[1:-1]
    for i in range(len(poly)):
        fill = extraire_valeur("fill",poly[i])
        points = extraire_valeur("points",poly[i])
        stroke = extraire_valeur("stroke",poly[i])
        stroke_width = extraire_valeur("stroke-width",poly[i])
        liste_de_points = [tuple(map(int, p.split(','))) for p in points.split()]
        liste_de_points+= [liste_de_points[0]]
        liste_de_points = [(x*2, y*2) for (x, y) in liste_de_points]
        for i in range(len(liste_de_points)-1):
            cv2.line(image, liste_de_points[i], liste_de_points[i+1], (255, 255, 255), thickness=int(float(stroke_width)*2))
        pts = np.array([liste_de_points], dtype=np.int32)
        cv2.fillPoly(image, pts, color=hex_to_rgb(fill)[::-1])
    cv2.imwrite(s_path, image)    
    return
    
@njit   
def seuillage_compact(img, seuil=124):
    arr=np.array(img,dtype=np.uint8)
    height,width=arr.shape[:2];
    new_arr=np.empty((height,width),np.float32)
    for h in range(height):
        for w in range(width):
            new_arr[h,w]=(0.2989*arr[h,w][0]+0.587*arr[h,w][1]+0.114*arr[h,w][2]>seuil)*255
    return Image.fromarray(new_arr.astype(np.uint8)).convert('1')

from dotenv import load_dotenv
load_dotenv()
#je met le token dans un .env pour la sécurité
bot.run(os.getenv("DISCORD_TOKEN"))




