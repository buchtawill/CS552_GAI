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
LOG_NAME = 'game_history.log'

LOC_NAMES = {
    "ak": "The Atwater Kent Electrical Engineering Building",
    "bar": "A laid back bar",
    "lab": "Wizard lab",
    "library": "The Library",
    "rome": "The Roman Colosseum",
}

GAME_TONES = ["Harry Potter", "1800s england", "epic japanese anime", "wild wild west"]

LOC_DESC = {
    "ak": "A sprawling, old building humming with the ghosts of innovation and forgotten experiments. Tesla coils can be heard buzzing around from time to time",
    "bar": "A bar with a relaxed atmosphere and a friendly bartender",
    "lab": "A chaotic yet strangely ordered chamber filled with bubbling potions, arcane symbols, and mystical artifacts",
    "library": "A vast repository of knowledge, with towering shelves of books and a hushed, ancient atmosphere",
    "rome": "The majestic ruins of a gladiatorial arena, strangely appearing as if plucked from the past"
}

def write_to_log(s:str, level=99):
    try:
        with open(LOG_NAME, 'a') as f:
            f.write(s+'\n')
            f.flush()
    except Exception as e:
        print(f"ERROR [game.py::write_to_log] Error writing to log: {e}")
    

def generate_initial_context()->str:

    context  = "Please run a game for me. The goal of the game is for the player (a human) to talk to multiple NPCs and find their way back home. \n" 
    context += "The player can win by entering a secret passphrase that must be retrieved from one of the NPCs. \n"
    context += "The point of the game is to be generative and novel every time it's played. You must decide ALL actions of the game. \n"
    context += "Each NPC knows who the passphrase holder is. Some of them may misdirect the player to the wrong NPC. \n"
    context += "The player should have to visit between 3-5 locations before unveiling the passphrase. Keep that in mind when running the game. \n"
    context += "Some NPCs may even lie about knowing the passphrase and what it is. It's up to you. But don't make the game too hard. \n"
    context += "ONLY ONE NPC KNOWS THE PASSPHRASE. To retrieve the passphrase, the player must convince the NPC to either give it, \n"
    context += "or solve a simple (easy) riddle to get a hint about who has it. \n"
    context += "If the player cannot solve the ridle in 2-3 tries, they can come back later to try a new one. \n"
    context += "The game should run for 3-5 rounds, where each round is a set of interactions with an NPC at a single location. \n"
    context += "If the player goes to the NPC with the passphrase at the first round, have the NPC misdirect them. \n"
    context += "Do not include any special characters in your response, just ASCII. \n"
    context += "The game has five set locations in addition to a home base which connects them all. These locations are: "
    context += "\n"
    context += "1st: " + LOC_NAMES["ak"]      + ": " + LOC_DESC["ak"]      + ". Short reference/dict key: 'ak'. \n"
    context += "2nd: " + LOC_NAMES["bar"]     + ": " + LOC_DESC["bar"]     + ". Short reference/dict key: 'bar'. \n"
    context += "3rd: " + LOC_NAMES["lab"]     + ": " + LOC_DESC["lab"]     + ". Short reference/dict key: 'lab'. \n"
    context += "4th: " + LOC_NAMES["library"] + ": " + LOC_DESC["library"] + ". Short reference/dict key: 'library'. \n"
    context += "5th: " + LOC_NAMES["rome"]    + ": " + LOC_DESC["rome"]    + ". Short reference/dict key: 'rome'. \n"
    context += "The locations are done being described."
    context += "Keep responses as short and concise as possible. Do not talk about the game, just respond to the prompt as-is, unless asked to do so otherwise."
    tone = random.choice(GAME_TONES)
    context += f"\nThe tone of the game is {tone}. Make sure to format your responses in a way that is hyper-formatted to the tone. "
    context += "End of context, start of instructions: \n\n"
    
    return context, tone

def get_game_setup(context)->dict:
    write_to_log("INFO [game.py::get_game_setup()] Requesting initial game state.")
    prompt = context + INIT_GAME_PROMPT
    write_to_log("INFO [game.py::get_game_setup()] Initial prompt: ")
    write_to_log(prompt)
        
    response = prompt_llm(prompt)
    
    write_to_log("INFO [game.py::get_game_setup()] Parsing game state response.")
    write_to_log("INFO [game.py::get_game_setup()] Initial raw LLM game setup: ")
    write_to_log(response + '\n')
    
    # Some models have "```json" as the first few characters. Use snapshot to keep formatting consistent
    game_setup_str = response
    
    # Start processing the initial game state
    game_setup = json.loads(game_setup_str)
    
    write_to_log(f"INFO [game.py::get_game_setup()] Tone of the game: {tone}")
    for loc in LOC_NAMES:
        write_to_log(f"INFO [game.py::get_game_setup()] Character name: {game_setup['characters'][loc]['name']} at {loc}")
        write_to_log(f"INFO [game.py::get_game_setup()] Character desc: {game_setup['characters'][loc]['description']}")
        
    write_to_log(f"INFO [game.py::get_game_setup()] Passphrase is   {game_setup['passphrase']}")
    write_to_log(f"INFO [game.py::get_game_setup()] Passphrase NPC: {game_setup['passphrase_holder']}")
    write_to_log(f"INFO [game.py::get_game_setup()] Context: {game_setup['context']}")
    
    return game_setup

def prompt_llm(prompt)->str:
    write_to_log("INFO [game.py] Prompting LLM: \n")
    write_to_log("------------------------- Start of prompt -------------------------")
    write_to_log(prompt)
    write_to_log("-------------------------- End of prompt --------------------------")
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
    Tell the LLM that it is playing a game, give the state, and a history of interactions. 
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
    instruction_base += '\n'

    return instruction_base

def get_next_travel_loc(state):
    """
    Get the next travel location and check for input errors
    """
    
    print(f"You can travel to the following locations, or home to guess the passphrase:")
    for loc in LOC_NAMES:
        if(loc != game_state['location']):
            print(f"- '{loc}': {LOC_NAMES[loc]} ")
            
    valid = False
    next_loc = input("Where would you like to go? ")
    while(not valid):
        valid = next_loc in LOC_NAMES or next_loc == 'home'
        if(valid):
            state["location"] = next_loc
        else:
            next_loc = input("Invalid location. Please enter a valid location: ")
            
    print()
    return next_loc

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

def play_round(context:str, setup:dict, state:dict, history:list)->list[dict, str]:
    '''
    Play a round of the game. This function will be called repeatedly until the game is over.
    A round consists of the player interacting with an NPC at a location any given number of times.
    The game is over when the player has found the passphrase and entered it correctly.
    params:
        context(str): The context of the game, a string to pass to the LLM.
        setup(dict): The game setup, a dictionary containing the names of characters, LLM context, passphrase, and passphrase holder.
        state(dict): The game state, a dictionary containing the current round and location.
    
    '''
    
    instruction_base = get_playtime_prompt(setup, state, history)
    
    # If at home base, choose a place to travel to. Otherwise, start NPC interaction
    if(state['location'] == 'home'):
        print(f"You are at home base.")
        next_loc = get_next_travel_loc(state)
        state["location"] = next_loc
        
        if(next_loc == "home"):
            return state, history
    
    npc_name = setup['characters'][state['location']]['name']
    print(f"You are now at {LOC_NAMES[state['location']]} and encounter {npc_name}.")
    # img = get_dalle_loc_img(setup, state)
    # plt.imshow(img)
    # plt.axis('off')  # Turn off axis labels
    # plt.title(f"Location: {LOC_NAMES[state['location']]}")
    # plt.show()
    
    # Prompt the LLM for the next response
    instruction = f"The player is at {state['location']} during round {state['round']}. Give the player the next logical thing "
    instruction += f"that {npc_name} would say given the current progression of the game. The player's name is {setup['name']}."
    instruction += " Simply continue off from where the conversation last ended - DO NOT INCLUDE ANY OTHER TEXT THAN WHAT THE NPC WOULD SAY. "
    # instruction += "Format your response as a narrator in a book. "
    prompt = context + instruction_base + instruction
    
    response = prompt_llm(prompt)
    slow_print(response)
    history.append(f"{npc_name} RESPONDED: " + response + " END OF NPC RESPONSE \n")
    
    keep_interacting = True
    while(keep_interacting):
        print()
        player_response = input(f"{setup['name']}, how do you respond? Enter a response, or 'travel' to move. ")
        print()
        if(player_response == 'travel'):
            next_loc = get_next_travel_loc(state)
            state["location"] = next_loc
            keep_interacting = False
        
        else:
            # Send a response to the LLM
            history.append(f"THE PLAYER RESPONDED TO {npc_name}: " + player_response + " END OF PLAYER RESPONSE.")
            prompt = context + get_playtime_prompt(setup, state, history) + instruction
            response = prompt_llm(prompt)
            slow_print(response)
            history.append("The game responded: " + response + " END OF GAME RESPONSE \n")
            
    state["round"] = state["round"] + 1
    return state, history

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

    context, tone = generate_initial_context()
    
    # Keys: 'characters'
    #           'name'
    #               'location'
    #               'description'
    #       'passphrase'
    #       'passphrase_holder'
    #       'context'
    game_setup = get_game_setup(context)
    game_setup['tone'] = tone
    game_state = {
        'round' : 0,
        'location' : 'home' # key to the LOC_NAMES dict
    }
    game_history = []
    
    # Start the game
    write_to_log("INFO [game.py] Starting game.")
    
    print("Welcome, traveler!")
    print(game_setup['context'])
    name = input("What is your name? ")
    name = name.capitalize()
    game_setup['name'] = name
    
    print(f"Hello, {name}")
    
    has_won = False
    while(not has_won):
        game_state, game_history = play_round(context, game_setup, game_state, game_history)
        
        if(game_state['location'] == 'home'):
            yorn = input("Would you like to guess the passphrase? (y/n): ")
            if(yorn == 'y'):
                guess = input("Enter your guess: ")
                if(guess.lower() == game_setup['passphrase'].lower()):
                    print("Yes! You win!!")
                    has_won = True
                else:
                    print("Nope! Better luck next time...")
        
    