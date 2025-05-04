import os
import time
import json
import random
from openai import OpenAI
import openai

import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

from prompt import *
from api_key import OPENAI_API_KEY as KEY

client = OpenAI(api_key=KEY)

proompt = "Do not include text in the image. you are a graphic designer for a video game. Generate an image that follows this environment: A vast frontier under stormy skies blends wild west saloons, Roman ruins, crackling laboratories, arcane lairs, and endless library rows into one dreamlike landscape. Strange inventions spark beside bubbling potions while outlaws and cowboys share drinks with wizards. Across the dusty paths and neon-lit windows, shadows drift and tumbleweeds roll, hinting that nothing here is ever what it first seems.. The theme of the game is wild wild westSpecifically, generate an image of the given location: lab (A chaotic yet strangely ordered chamber filled with bubbling potions, arcane symbols, and mystical artifacts) under these constraints."

response = client.images.generate(
    model="dall-e-3",               # Or "dall-e-2" if using that
    prompt=proompt,
    size="1024x1024",
    n=1
)

image_url = response.data[0].url
print("Image URL:", image_url)

# Download the image
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

plt.imshow(image)
plt.axis('off')  # Turn off axis labels
plt.show()