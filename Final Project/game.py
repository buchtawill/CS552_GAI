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

from prompt import * # A set of predefined prompts
from api_key import OPENAI_API_KEY as KEY

# MODEL_SNAPSHOT = "gpt-4.1-nano-2025-04-14"
MODEL_SNAPSHOT = "gpt-4.1-2025-04-14"
LOG_NAME = 'gamelog.log'

GAME_TONES = ["Harry Potter", "1800s england", "epic japanese anime", "wild wild west", "cyberpunk, sci-fi, yet dystopian"]

def write_to_log(s:str, level=99):
    try:
        with open(LOG_NAME, 'a') as f:
            f.write(s+'\n')
            f.flush()
    except Exception as e:
        print(f"ERROR [game.py::write_to_log] Error writing to log: {e}")
    

def generate_initial_premise()->str:

    tone = random.choice(GAME_TONES)
    context = GAME_CONTEXT + f"The tone of the game is {tone}. Make sure to format your responses in a way that is exaggerated to the tone.\n"
    context += "END OF CONTEXT. START OF INSTRUCTIONS. \n\n"
    
    return context, tone

def get_dicts_from_json(json: dict):
    """
    Parse the json format specified in the INIT_GAME_PROMPT. Create two new dicts that the game can use, one with a list of locations
    and one with a list of NPCs.
    
    args:
        json(dict): The json object to parse.
    
    returns:
        locations(dict): A dictionary of locations, with natural numbers as the key to each location
        npcs(dict): A dictionary of NPCs, with the location reference as the key to a sub dict
    
    """
    
    # Have NPC name and want to find location? npcs
    # Have location and want to find npc name? --> locs[location]['npc_name']
    
    locs = {}
    npcs = {}
    
    n_locs = json['n_locations']
    for i in range(n_locs):
        current_loc_dict = json['locations'][i]
        loc_name = current_loc_dict['name']
        loc_npc_name = json['npcs_by_loc'][loc_name]['npc_name']
        
        locs[loc_name] = {
            "ref": current_loc_dict['ref'],
            "desc": current_loc_dict['desc'],
            "relevance": current_loc_dict['theme_relevance'],
            "npc_name": json['npcs_by_loc'][loc_name]['npc_name']
        }
        
        # Duplicate the npcs dict to have the name of location as the key for ease of access
        npcs[loc_npc_name] = {
            "npc_desc": json['npcs_by_loc'][loc_name]['npc_desc'],
            "npc_location": loc_name
        }

    return locs, npcs
    

def get_game_setup(context)->dict:
    """
    Do the initial prompt to the LLM to get the locations and characters set up.
    """
    
    write_to_log("INFO [game.py::get_game_setup()] Requesting initial game state.")
    prompt = context + INIT_GAME_PROMPT
    # write_to_log("INFO [game.py::get_game_setup()] Initial prompt: ")
    # write_to_log(prompt)
    
    response = prompt_llm(prompt)
    
    write_to_log("INFO [game.py::get_game_setup()] Parsing game state response.")
    
    # Start processing the initial game state
    try:
        game_setup = json.loads(response)
    except Exception as e:
        write_to_log("ERROR [game.py::get_game_setup()] Error parsing game state response: " + str(e))
        exit()
    
    locations, npcs = get_dicts_from_json(game_setup)
    
    for loc in locations.keys():
        name = loc
        desc = locations[loc]['desc']
        npc = locations[loc]['npc_name']
        npc_desc = npcs[npc]['npc_desc']
        
        write_to_log(f"INFO [game.py::get_game_setup()] {name}: {desc}")
        write_to_log(f"INFO [game.py::get_game_setup()]   - NPC at {name}: {npc}. {npc_desc}")
        
    write_to_log(f"INFO [game.py::get_game_setup()] Passphrase is   {game_setup['passphrase']}")
    write_to_log(f"INFO [game.py::get_game_setup()] Passphrase NPC: {game_setup['passphrase_holder']}")
    write_to_log(f"INFO [game.py::get_game_setup()] Context: {game_setup['context']}")
    
    return game_setup, locations, npcs

def prompt_llm(prompt)->str:
    """
    A wrapper around the API call to openai
    """
    
    write_to_log("INFO [game.py] Prompting LLM:")
    write_to_log("INFO [game.py] ------------------------- Start of prompt -------------------------")
    write_to_log(prompt)
    write_to_log("INFO [game.py] -------------------------- End of prompt --------------------------")
    response = client.responses.create(
        model = MODEL_SNAPSHOT,
        input=prompt
    )
    if(response.error):
        write_to_log("ERROR [game.py::get_game_setup()] OpenAI API error: " + response.error)
        exit()
    
    write_to_log(response.output_text)
        
    return response.output_text

def get_playtime_prompt(setup:dict, state:dict, history:list):
    """
    Tell the LLM that it is now playing the game, give the current state, and a history of interactions. 
    """
    # Setup the prompt base for the LLM
    instruction_base = "You are now playing the game. Here is a summary of the characters, their locations and personalities, "
    instruction_base += "as well as the passphrase to winning the game and who holds that information: \n"
    instruction_base += json.dumps(setup, indent=4)
    instruction_base += f"This is the current state of the game: \n{json.dumps(state, indent=4)}\n"
    if(len(history) > 0):
        instruction_base += "This is a history of interactions the game has done so far: "
        for i in history:
            instruction_base += i + '\n'
        instruction_base += "END OF HISTORY\n"
    instruction_base += '\n'
    

    return instruction_base

def check_shorthand_valid(shorthand:str, locations)->bool:
    """
    Check if the given shorthand location is valid. If it is, return true and the name of the location
    """
    for loc in locations:
        if shorthand == locations[loc]['ref']:
            return True, loc
    return False, None

def get_next_travel_loc(master_dict:dict)->str:
    """
    Get the next travel location and check for input errors
    """
    
    loc_dict = master_dict['locations']
    
    print(f"You can travel to the following locations, or home to guess the passphrase:")
    for loc in loc_dict:
        if(loc != master_dict['state']['location']):
            print(f"- {loc_dict[loc]['ref']}: {loc}")            
    print(f"- 'home' (guess the passphrase)")
            
    valid = False
    next_loc = input("Where would you like to go? ")
    while(not valid):
        valid, loc_name = check_shorthand_valid(next_loc, master_dict['locations']) 
    
        if(valid or (next_loc == 'home')):
            if(next_loc == 'home'):
                return 'home'
            else:
                return loc_name
        else:
            next_loc = input("Invalid location. Please enter a valid location: ")
            
    return None

def slow_print(s:str, delay=0.02):
    for char in s:
        print(char, end='', flush=True)
        time.sleep(delay)

def get_dalle_loc_img(setup:dict, state:dict)->Image:
    
    proompt = f"you are a graphic designer for a video game. Generate an image that follows this environment: {setup['environment']}. "
    proompt += f"The theme of the game is {setup['tone']}. Do not include text in the image. "
    proompt += f"Specifically, generate an image of the given location: {state["location"]} ({LOC_DESC[state['location']]}) under these constraints."

    write_to_log(f"Getting image from Dalle. Prompt: {proompt}")
    response = client.images.generate(
        model="dall-e-2",    # dall e 2 is cheaper
        prompt=proompt,
        size="1024x1024",
        n=1,
    )
    
    image_url = response.data[0].url
    write_to_log(f"Image URL: {image_url}")

    # Download the image
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    return image

def play_round(master_dict:dict)->dict:
    '''
    Play a round of the game. This function will be called repeatedly until the game is over.
    A round consists of the player interacting with an NPC at a location any given number of times.
    The game is over when the player has found the passphrase and entered it correctly.
    params:
        master_dict(dict): A dictionary containing all information necessary to keep the game running.
    
    '''
    
    # If at home base, choose a place to travel to. Otherwise, start NPC interaction
    if(master_dict['state']['location'] == 'home'):
        print(f"You are at home base.")
        next_loc = get_next_travel_loc(master_dict)
        master_dict['state']['location'] = next_loc
        
        # Go back home to guess
        if(next_loc == "home"):
            return master_dict
    
    current_loc = master_dict['state']['location']
    curr_round = master_dict['state']['round']
    npc_name = master_dict['locations'][current_loc]['npc_name']
    print(f"You are now at {current_loc} and encounter {npc_name}.")
    
    # img = get_dalle_loc_img(setup, state)
    # plt.imshow(img)
    # plt.axis('off')  # Turn off axis labels
    # plt.title(f"Location: {LOC_NAMES[state['location']]}")
    # plt.show()
    
    # Prompt the LLM for the next response
    instruction_base = get_playtime_prompt(master_dict['setup'], master_dict['state'], master_dict['history'])
    
    instruction = f"The player is at {current_loc} during round {curr_round}. Give the player the next logical thing "
    instruction += f"that {npc_name} would say given the current progression of the game. The player's name is {master_dict['setup']['name']}."
    instruction += " Simply continue off from where the conversation last ended - DO NOT INCLUDE ANY OTHER TEXT THAN WHAT THE NPC WOULD SAY. "
    # instruction += "Format your response as a narrator in a book. "
    prompt = master_dict['llm_premise'] + instruction_base + instruction
    
    response = prompt_llm(prompt)
    slow_print(response)
    hist = master_dict['history']
    hist.append(f"{npc_name} RESPONDED: " + response + " END OF NPC RESPONSE \n")
    master_dict['history'] = hist
    
    keep_interacting = True
    while(keep_interacting):
        print()
        player_response = input(f"{master_dict['setup']['name']}, how do you respond? Enter a response, or 'travel' to move. ")
        print()
        if(player_response == 'travel'):
            next_loc = get_next_travel_loc(master_dict)
            master_dict['state']['location'] = next_loc
            keep_interacting = False
        
        else:
            # Send a response to the LLM
            hist = master_dict['history']
            hist.append(f"THE PLAYER RESPONDED TO {npc_name}: " + player_response + " END OF PLAYER RESPONSE.")
            master_dict['history'] = hist
            
            prompt = master_dict['llm_premise'] + get_playtime_prompt(master_dict['setup'], master_dict['state'], master_dict['history']) + instruction
            response = prompt_llm(prompt)
            slow_print(response)
            
            hist = master_dict['history']
            hist.append(f"{npc_name} RESPONDED: " + response + " END OF NPC RESPONSE \n")
            master_dict['history'] = hist
            
    master_dict['state']['round'] = curr_round + 1
    return master_dict

if __name__ =='__main__':
    
    # delete the log file if it exists
    if os.path.exists(LOG_NAME):
        os.remove(LOG_NAME)
    # create a new log file
    with open(LOG_NAME, 'w') as f:
        f.write("Game log\n")
        f.write("========\n")
    
    client = OpenAI(api_key=KEY)
    # l = client.models.list()
    # for m in l:
        # print(m)
    # exit()    

    premise, tone = generate_initial_premise()
    
    # game_setup can be fed back into an LLM to provide context of the current game
    # locations and npcs dicts are meant to be used by the game engine.
    game_setup, locations, npcs = get_game_setup(premise)
    
    game_state = {
        'round' : 0,
        'location' : 'home'
    }
    master_dict = {
        'llm_premise':premise,      # Premise of what the LLM is doing
        'setup':game_setup,         # context of the game for the LLM
        'state': game_state,        # 'round' and 'location'
        'history': [],              # List of previous interactions
        'npcs': npcs,               # Dict of NPCs
        'locations': locations      # Dict of locations
    }
    
    write_to_log("############################### setup ###############################")
    write_to_log(json.dumps(master_dict['setup'], indent=4))
    write_to_log("############################### NPCS ############################### ")
    write_to_log(json.dumps(master_dict['npcs'], indent=4))
    write_to_log("############################### Locations ###############################")
    write_to_log(json.dumps(master_dict['locations'], indent=4))
    
    
    # Start the game
    write_to_log("INFO [game.py] Starting game.")
    
    print("Welcome!")
    print(game_setup['context'])
    name = input("What is your name? ")
    name = name.capitalize()
    game_setup['name'] = name
    
    print(f"Hello, {name}")
    has_won = False
    while(not has_won):
        master_dict = play_round(master_dict)
        
        if(game_state['location'] == 'home'):
            yorn = input("Would you like to guess the passphrase? (y/n): ")
            if(yorn == 'y'):
                guess = input("Enter your guess: ")
                if(guess.lower() == game_setup['passphrase'].lower()):
                    print("Yes! You win!!")
                    has_won = True
                else:
                    print("Nope! Better luck next time...")
        
    