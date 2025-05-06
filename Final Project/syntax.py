import os
import time
import json
import random
from openai import OpenAI
import openai
from api_calls import *

import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

from PIL import Image

from prompt import *
from api_key import OPENAI_API_KEY as KEY

client = OpenAI(api_key=KEY)

prompt = "Generate a wild-wild-west texture that can be tiled. It should be a good background, and not too heavy on details."


prompt = "A slender swordsman, pale blue kimono trailing, feet barely touching lotus leaves on the water, black hair like ink."
img = get_pixel_art(prompt=prompt, size=(64,128), style='portrait', b_tile=False, rm_bg=True)

img = Image.open(img)
img.show()