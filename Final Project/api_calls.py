import json
import base64
import requests
from PIL import Image
from io import BytesIO
from datetime import datetime

import openai
from api_key import *
from openai import OpenAI

from log import write_to_log

# MODEL_SNAPSHOT = "gpt-4.1-nano-2025-04-14"
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Valid styles to be sent to retro ai
valid_retro_diff_styles=[
    "default",
    "retro",
    "simple",
    "detailed",
    "anime",
    "game_asset",
    "portrait",
    "texture",
    "ui",
    "item_sheet",
    "mc_texture",
    "mc_item",
    "character_turnaround",
    "1_bit",
    "animation_four_angle_walking",
    "no_style"
]

def get_pixel_art(prompt:str, size:tuple, style:str, b_tile:bool, rm_bg:bool=False, input_b64:str=None)->tuple[str, str]:
    """
    Get a pixel art image from retro diffusion AI.
    params: 
        prompt(str): A prompt to generate from.
        size(tuple): An (int, int) tuple of pixel sizes
        style(str): The style of the prompt to use. Available options
            are listed in api_calls.py
        tile(bool): Whether or not this should be tiled
        rm_bg(bool): Whether or not to remove the background.
        input_b64(str): Optional: An input image for the model to work with. 
        
    Returns:
        A filename (path) to where the image is saved.
        base64 (str): The image in base64.
    """
    
    if style not in valid_retro_diff_styles:
        raise Exception(f"ERROR [api_calls.py::get_pixel_art()] Invalid prompt style {style}")
    
    if(type(size[0]) is not int or type(size[1]) is not int):
        raise Exception(f"ERROR [api_calls.py::get_pixel_art()] Invalid image size: {size}")
        
    
    url = "https://api.retrodiffusion.ai/v1/inferences"
    method = "POST"

    headers = {
        "X-RD-Token": RETRO_DIFFUSION_API_KEY,
    }

    payload = {
        "model": "RD_FLUX",
        "width": size[0],
        "height": size[1],
        "prompt": prompt,
        "num_images": 1,
        "prompt_style": style,
        "tile_x": b_tile,
        "tile_y": b_tile,
        "remove_bg": rm_bg
    }
    
    if(input_b64):
        payload['input_image'] = input_b64
    
    write_to_log("INFO [api_calls.py::get_pixel_art()] Sending request to retro diffusion:")
    write_to_log("INFO [api_calls.py::get_pixel_art()] ------------------------- Start of payload -------------------------")
    write_to_log(json.dumps(payload, indent=4))
    write_to_log("INFO [api_calls.py::get_pixel_art()] -------------------------- End of payload --------------------------")
    
    
    try:
        response = requests.request(method, url, headers=headers, json=payload)
        response.raise_for_status()  # Raises HTTPError for bad HTTP status codes
    except requests.exceptions.RequestException as e:
        write_to_log(f"ERROR [api_calls.py::get_pixel_art()] HTTP request failed: {e}")
        raise  # Or return None, or handle however you'd like
    
    # Response format
    # {
    #     "created_at": 1733425519,
    #     "credit_cost": 1,
    #     "base64_images": ["..."],
    #     "model": "RDModel.RD_FLUX",
    #     "type": "txt2img",
    #     "remaining_credits": 999
    # }
    
    write_to_log(f"INFO [api_calls.py::get_pixel_art()] Response from retro diffusion: {response.text}")

    # Now check if the API returned a failure message in the JSON
    try:
        j = response.json()
        if "detail" in j:
            write_to_log(f"ERROR [api_calls.py::get_pixel_art()] API error response: {j['detail']}")
            raise Exception(f"Inference failed: {j['detail']}")
    except ValueError as e:
        write_to_log(f"ERROR [api_calls.py::get_pixel_art()] Failed to decode JSON: {e}")
        raise
    
    image_data = base64.b64decode(j['base64_images'][0])
    image = Image.open(BytesIO(image_data))

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = f"resources/pixelart_{timestamp}.png"

    image.save(filename)
    
    return filename, j['base64_images'][0]

def prompt_llm(prompt)->str:
    """
    A wrapper around the API call to openai. This function will write the prompt and response
    """        
    
    write_to_log("INFO [game.py] Prompting LLM:")
    write_to_log("INFO [game.py] ------------------------- Start of prompt -------------------------")
    write_to_log(prompt)
    write_to_log("INFO [game.py] -------------------------- End of prompt --------------------------")
    
    try:
        response = openai_client.responses.create(
            model = OPENAI_MODEL_SNAPSHOT,
            input=prompt
        )
        
        if(response.error):
            write_to_log("ERROR [game.py::get_game_setup()] OpenAI API error: " + response.error)
            raise Exception("OpenAI API error")

        write_to_log("INFO [game.py] ------------------------- Start of Response -------------------------")
        write_to_log(response.output_text)
        write_to_log("INFO [game.py] ------------------------- End of Response -------------------------")
        
        return response.output_text
    
    except Exception as e:
        write_to_log(f"ERROR [game.py::get_game_setup()] OpenAI API error: {e}")
        raise
        
    
# def get_dalle_loc_img(main)->Image:
    
#     tone = main['setup']['theme']
#     loc = main['state']['location']
#     desc = main['locations'][loc]['desc']
#     npc_name = master_dict['locations'][loc]['npc_name']
#     npc_desc = master_dict['npcs'][npc_name]['npc_desc']
    
#     proompt = f"you are a graphic designer for a video game. Generate an image that follows this environment: {main['setup']['environment']}. "
#     proompt += f"The theme of the game is {tone}. Do not include text in the image. "
#     proompt += f"Specifically, generate an image of the given location: {loc} ({desc}). In the image, include the character: "
#     proompt += f"{npc_name}: {npc_desc}. "

#     write_to_log(f"Getting image from Dalle. Prompt: {proompt}")
#     response = client.images.generate(
#         model="dall-e-3",    # dall e 2 is cheaper
#         prompt=proompt,
#         size="1024x1024",
#         n=1,
#     )
    
#     image_url = response.data[0].url
#     write_to_log(f"Image URL: {image_url}")

#     # Download the image
#     response = requests.get(image_url)
#     image = Image.open(BytesIO(response.content))

#     return image