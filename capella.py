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
#|-> resize images to match braille structure
#>>> transform images in binary, with a floyd arror diffusion system
#>>> cut images on small images and convert them on braille character

@bot.tree.command(name="ascii_art", description="met ton image en ascii !", guild=None)
@app_commands.describe(width="choisir la taille (max 100)",inverted="couleur inversé ?",)
async def image_command(interaction: discord.Interaction,height: int,attachment: discord.Attachment, inverted: bool=False):
    await interaction.response.defer(thinking=True)
    inverted=not inverted
    if not attachment.filename.lower().endswith((".png", ".jpg", ".bmp")):
        await interaction.followup.send("l'image doit etre du format .png, .jpg ou .bmp")
        return
    if height>100:
        await interaction.response.send_message("hopopop ! pas plus de 100 !")
        return
    path=f"temp/{attachment.filename}"
    await attachment.save(path)
    liste=asciiator(path, height, invert, resize=True)
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
    
def floyd_error_diff(image):
    array=np.array(image.convert('L'), dtype=float)
    min_val=array.min()
    max_val=array.max()
    if max_val>min_val:
        array=(array-min_val)*(255/(max_val-min_val))
    height, width=array.shape
    for i in range(height-1):
        for j in range(width-1):
            old_pixel=array[i,j]
            new_pixel=255 if old_pixel>=128 else 0
            error=old_pixel-new_pixel
            array[i,j]=new_pixel
            if j+1<width:
                array[i,j+1]+=error*0.4375
            if i+1<height and j>0:
                array[i+1,j-1]+=error*0.1875
            if i+1<height:
                array[i+1,j]+=error*0.3125
            if i+1<height and j+1<width:
                array[i+1,j+1]+=error*0.0625
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))

def resizer(wh):
    w=wh[0]
    h=wh[1]
    ratio=5*w/(3*h)
    new_h=4*h
    new_w=2*h*ratio
    return (new_w,new_h)
def asciiator(path, lign, invert, resize=True):
    i=Image.open(path)
    if resize:
        w,h=resizer(i.size)
    else:
        w,h=i.size
    width=int(w*lign*4/h)
    i=i.resize((width,lign*4), Image.LANCZOS)
    i=i.convert('L')
    i=floyd_error_diff(i)
    li=[]
    for lgn in range(0,lign*4,4):
        li2=[]
        for col in range(0,i.size[0],2):    
            box=(col, lgn, col+2, lgn+4)
            sub=i.crop(box)
            st=""
            for x in range(2):
                for y in range(4):
                    st+=str(0 if sub.getpixel((x,y)) == 0 else 1)
            st=st[7]+st[6]+st[5]+st[2]+st[4]+st[1]+st[3]+st[0]
            if not invert:
                st=st.replace('1','2')
                st=st.replace('0','1')
                st=st.replace('2','0')
            hexa=int(st, 2)
            li2.append('⠠' if hexa==0 else chr(hexa+10240))
        li.append(li2)
    return li

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





