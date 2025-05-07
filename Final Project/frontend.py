import sys
import math
import time
import pygame
from api_calls import *

import random
# from game import *
import game

import pickle
import threading

import os
import log 
if os.path.exists(log.LOG_NAME):
    os.remove(log.LOG_NAME)


# --- Constants ---
TILE_SIZE = 64
BG_SPRITE_SIZE = (int(TILE_SIZE*2/3), int(TILE_SIZE*2/3))

MINIMAP_WIDTH = 150
MINIMAP_HEIGHT = 60
MINIMAP_POS = (10, 10)

NPC_SIZE = (64, 128)
NPC_SIZE_SCALED = (128, 256)
PLAYER_SIZE = (64, 128)
player_speed = 8

WORLD_SIZE_TILES = (48, 20)
WORLD_SIZE_PIX = (WORLD_SIZE_TILES[0] * TILE_SIZE, WORLD_SIZE_TILES[1] * TILE_SIZE)
base_res = (TILE_SIZE*16, TILE_SIZE*10) # make it a multiple of tile size

LOC_SIZE = (base_res[0] // 2, base_res[1] // 2)
LOC_ICON_SIZE = (int(TILE_SIZE * 3), int(TILE_SIZE * 3))

zoom = 1
window_size = (base_res[0] * zoom, base_res[1] * zoom)

S_MAIN_WORLD       = 0
S_IN_LOCATION      = 1
S_GET_PLAYER_NAME  = 2
S_PRINT_GAME_INFO  = 3
S_GET_PLAYER_INPUT = 4

S_NPC_PROMPT_LLM    = 0
S_NPC_TALK          = 1
S_PLAYER_TURN       = 2
S_WAIT_READ         = 3 # Wait for 2-3 seconds for the player to read the text 

# Number of icons to place in the world
NUM_BG_SPRITES = random.randint(15, 30)  # Random number between 5 and 15 icons

# Create a list to store icon positions (as Rects or tuples)
bg_sprite_pos = []


loc_icon_pos = [
    (512, 320),  # Top left
    (1536, 320), # Top mid
    (2560, 320), # Top right
    (512, 960),  # bottom left
    (2560, 960)  # Bottom right
]

def initialize_icons():
    
    write_to_log(f"INFO [frontend.py::initialize_icons()] Number of bg sprites: {NUM_BG_SPRITES}")
    for _ in range(NUM_BG_SPRITES):
        # Random x and y positions within the world size
        x_pos = random.randint(0, WORLD_SIZE_PIX[0] - BG_SPRITE_SIZE[0])
        y_pos = random.randint(0, WORLD_SIZE_PIX[1] - BG_SPRITE_SIZE[1])
        bg_sprite_pos.append((x_pos, y_pos))  # Store positions as tuples
        
# def update_icons():
#     for icon in bg_sprite_pos:
#         icon[0] += 

def get_game_art(game_setup:dict, locs:dict, npcs:dict):
    """
    Given the game setup, containing information regarding each location, get pixel art for
    the main character, background tile + 1 special background sprite,
    one for each location's interior, one for each location's icon on the main map.
    
    The base resolution of the canvas is 768 x 640. Thus, internal locations should be 384x320 pix
    
    All images in the return dict are converted to pygame textures
    
    5 location interiors, 512x320, 3 credits each
    5 loc icons, 128x128, 1 credit each
    5 NPCs, 48x96, 1 credit each
    
    1 background tile, TILE_SIZE x TILE_SIZE, 1 credit
    1 special sprite, half of tile size, 1 credit
    1 player sprite, PLAYER_SIZE[0] x PLAYER_SIZE[1]
    
    28 total credits / ~30 cents per function call :(
        
    TODO: Use threading to get all at the same time.
    
    Args:
        game_setup(dict): The game setup dict created by chat GPT
        locs(dict): A dictionary containing information and prompts about locations
        npcs(dict): A dictionary containing informatino and prompts about NPCs
        
    Returns:
        all_tiles: a dict with the following structure:
            {
                "<loc name>" : {
                    "full" : <pygame image>,
                    "icon" : <pygame image>,
                    "full_path" : path to the image,
                    "icon_path" : path to the icon
                },
                ...
                <repeated for all locations>
                ...
                
                "<npc name>" : (<pygame image of that npc>, path to image),
                ...
                
                "player" : <pygame image of the player>,
                "player_path" : path to the img of the player,
                
                "floor_tile" : <pygame image of the floor tile>,
                
                "wall_tile" : <pygame image of a wall tile>,
                
                "bgnd_sprite" : <pygame image of the extra bg sprite>
            },
        
        all_paths: a dict with the same structure as all_tiles, but with the paths to the images instead of the pygame surfaces.
    
    """
    
    all_tiles = {}
    all_paths = {}
    
    # For every location, generate an icon, a full location, and the NPC
    for loc_name in locs:
        
        print(f"INFO [frontend.py::get_game_art()] Requesting textures for {loc_name}")
        
        write_to_log(f"INFO [frontend.py::get_game_art()] Getting images for location '{loc_name}'")
        
        loc_desc = locs[loc_name]['pixelart_prompt']
        loc_icon_desc = locs[loc_name]['pixelart_icon']
        
        npc_name = locs[loc_name]['npc_name']
        npc_desc = npcs[npc_name]['npc_art_prompt']
        
        npc_path, _ = get_pixel_art(npc_desc, size=NPC_SIZE, style="portrait", b_tile=False, rm_bg=True)
        loc_full_path, loc_b64 = get_pixel_art(loc_desc, size=LOC_SIZE, style="texture", b_tile=False, rm_bg=False)
        
        # Icon should be prompted as a game asset, transparent
        # Send the location image in the prompt with input image
        loc_icon_path, _ = get_pixel_art(loc_icon_desc, size=LOC_ICON_SIZE, style="game_asset", b_tile=False, rm_bg=False, input_b64=loc_b64)
        
        # Get all three
        all_tiles[loc_name] = {}
        all_paths[loc_name] = {}
        all_tiles[loc_name]['full'] = pygame.image.load(loc_full_path).convert_alpha()
        all_tiles[loc_name]['icon'] = pygame.image.load(loc_icon_path).convert_alpha()
        all_tiles[npc_name] = pygame.image.load(npc_path).convert_alpha()
        
        all_paths[loc_name]['full_path'] = loc_full_path
        all_paths[loc_name]['icon_path'] = loc_icon_path
        all_paths[npc_name] = npc_path
    
    # Generate background floor tile
    write_to_log(f"INFO [frontend.py::get_game_art()] Getting background floor tile")
    prompt = f"A very lightly-detailed tile of a floor with the theme {game_setup['theme']}. The color palette is light. There is very little information in this tile."
    floor_tile_path, floor_b64 = get_pixel_art(prompt, size=(TILE_SIZE, TILE_SIZE), style="texture", b_tile=True, rm_bg=False)
    
    # Generate wall tile for world boundary
    write_to_log(f"INFO [frontend.py::get_game_art()] Getting wall tile")
    prompt = f"A Wall tile according to the theme {game_setup['theme']}. This is a world boundary for the game, and should be dense with lots of detail. The color palette is dark."
    wall_tile_path, _ = get_pixel_art(prompt, size=(TILE_SIZE, TILE_SIZE), style="texture", b_tile=True, rm_bg=False, input_b64=floor_b64)
    
    # Generate background sprite
    write_to_log(f"INFO [frontend.py::get_game_art()] Getting background sprite")
    prompt = game_setup['special_sprite']
    bgnd_sprite_path, _ = get_pixel_art(prompt, size=BG_SPRITE_SIZE, style="game_asset", b_tile=False, rm_bg=True)
    
    # Generate player sprite
    write_to_log(f"INFO [frontend.py::get_game_art()] Getting player sprite")
    prompt = game_setup['player_art_prompt']
    player_path, _ = get_pixel_art(prompt, size=PLAYER_SIZE, style="portrait", b_tile=False, rm_bg=True)
    
    # Get Home icon
    write_to_log(f"INFO [frontend.py::get_game_art()] Getting home icon")
    prompt = game_setup['home_icon']
    home_path, _ = get_pixel_art(prompt, size=LOC_ICON_SIZE, style="game_asset", b_tile=False, rm_bg=True)
    
    # Save all tiles
    all_tiles['player'] = pygame.image.load(player_path).convert_alpha()
    all_tiles['home_icon'] = pygame.image.load(home_path).convert_alpha()
    all_tiles['wall_tile'] = pygame.image.load(wall_tile_path).convert_alpha()
    all_tiles['floor_tile'] = pygame.image.load(floor_tile_path).convert_alpha()
    all_tiles['bgnd_sprite'] = pygame.image.load(bgnd_sprite_path).convert_alpha()
    
    all_paths['player'] = player_path
    all_paths['home_icon'] = home_path
    all_paths['wall_tile'] = wall_tile_path
    all_paths['floor_tile'] = floor_tile_path
    all_paths['bgnd_sprite'] = bgnd_sprite_path
    
    return all_tiles, all_paths


def load_assets_from_paths(art_paths:dict):
    """
    Given a dictionary of paths to images, load them into pygame surfaces.
    
    Args:
        art_paths(dict): A dictionary with the same structure as the one returned by get_game_art()
        
    Returns:
        art_assets(dict): A dictionary with the same structure as art_paths, but with pygame surfaces instead of paths.
    """
    
    art_assets = {}
    
    for key in art_paths:
        if isinstance(art_paths[key], dict):
            art_assets[key] = {}
            for subkey in art_paths[key]:
                new_key = subkey
                if '_path' in subkey:
                    new_key = new_key.replace('_path', '')
                art_assets[key][new_key] = pygame.image.load(art_paths[key][subkey]).convert_alpha()
        else:
            new_key = key
            if '_path' in new_key:
                new_key = new_key.replace('_path', '')
            art_assets[new_key] = pygame.image.load(art_paths[key]).convert_alpha()
    
    return art_assets
    
def handle_events(events:pygame.event, render_state):
    for event in events:
        if event.type == pygame.QUIT:
            return False
        
    keys = pygame.key.get_pressed()
    if keys[pygame.K_ESCAPE]:
        return False
    if keys[pygame.K_q] and ((render_state != S_GET_PLAYER_NAME) and (render_state != S_NPC_TALK)):
        return False
    return True


def handle_movement():
    global player_world_pos
    keys = pygame.key.get_pressed()
    move = pygame.Vector2(0, 0)
    if keys[pygame.K_w]: move.y -= player_speed
    if keys[pygame.K_s]: move.y += player_speed
    if keys[pygame.K_a]: move.x -= player_speed
    if keys[pygame.K_d]: move.x += player_speed
    player_world_pos += move

def clamp_player_to_world():
    player_world_pos.x = max(0, min(player_world_pos.x, WORLD_SIZE_PIX[0] - PLAYER_SIZE[0]))
    player_world_pos.y = max(0, min(player_world_pos.y, WORLD_SIZE_PIX[1] - PLAYER_SIZE[1]))

def scroll_camera_to_player():
    global camera_offset
    player_screen_x = player_world_pos.x - camera_offset.x
    player_screen_y = player_world_pos.y - camera_offset.y

    if player_screen_x < scroll_margin.left:
        camera_offset.x -= scroll_margin.left - player_screen_x
    elif player_screen_x > scroll_margin.right:
        camera_offset.x += player_screen_x - scroll_margin.right

    if player_screen_y < scroll_margin.top:
        camera_offset.y -= scroll_margin.top - player_screen_y
    elif player_screen_y > scroll_margin.bottom:
        camera_offset.y += player_screen_y - scroll_margin.bottom

def draw_background():
    rows = math.ceil(base_res[1] / TILE_SIZE) + 2
    cols = math.ceil(base_res[0] / TILE_SIZE) + 2
    start_x = int(camera_offset.x // TILE_SIZE)
    start_y = int(camera_offset.y // TILE_SIZE)

    for row in range(start_y, start_y + rows):
        for col in range(start_x, start_x + cols):
            world_x = col * TILE_SIZE
            world_y = row * TILE_SIZE
            screen_x = world_x - camera_offset.x
            screen_y = world_y - camera_offset.y

            if 0 <= col < WORLD_SIZE_TILES[0] and 0 <= row < WORLD_SIZE_TILES[1]:
                tile = art_assets['floor_tile']
            else:
                tile = art_assets['wall_tile']
            game_surface.blit(tile, (screen_x, screen_y))

def draw_player(override_pos:tuple=None, scaled_size:tuple=None):
    """
    Draw the player in the world. If override is given, draw the player there instead. If the scaled size is given, resize the player
    
    params:
        override_pos(tuple) Optional: The screen coordinate to override the player's position (original position still preserved)
        scaled_size(tuple) Optional: The size to rescale the player to
        
    Returns:
        None
    """
    if(scaled_size): sz = scaled_size
    else:            sz = PLAYER_SIZE
    
    if(override_pos):
        
        player_rect = pygame.Rect(
            override_pos[0],
            override_pos[1],
            sz[0], sz[1]
        )
    else: 
        player_rect = pygame.Rect(
            player_world_pos.x - camera_offset.x,
            player_world_pos.y - camera_offset.y,
            sz[0], sz[1]
        )
        
    if(scaled_size):
        scaled_player = pygame.transform.scale(art_assets['player'], sz)
        game_surface.blit(scaled_player, player_rect)
    else:
        game_surface.blit(art_assets['player'], player_rect)

def draw_minimap():
    minimap_surface = pygame.Surface((MINIMAP_WIDTH, MINIMAP_HEIGHT), pygame.SRCALPHA)
    minimap_surface.fill((40, 40, 40, 180))  # Transparent dark gray
    game_surface.blit(minimap_surface, MINIMAP_POS)

    scale_x = MINIMAP_WIDTH / WORLD_SIZE_PIX[0]
    scale_y = MINIMAP_HEIGHT / WORLD_SIZE_PIX[1]

    # Draw player as red dot
    player_x = int(MINIMAP_POS[0] + player_world_pos.x * scale_x)
    player_y = int(MINIMAP_POS[1] + player_world_pos.y * scale_y)
    pygame.draw.circle(game_surface, (255, 0, 0), (player_x, player_y), 2)

    # Draw camera view as white rectangle
    camera_rect = pygame.Rect(
        MINIMAP_POS[0] + camera_offset.x * scale_x,
        MINIMAP_POS[1] + camera_offset.y * scale_y,
        base_res[0] * scale_x,
        base_res[1] * scale_y
    )
    pygame.draw.rect(game_surface, (255, 255, 255), camera_rect, 1)

def render_frame():
    scaled_surface = pygame.transform.scale(game_surface, window_size)
    screen.blit(scaled_surface, (0, 0))
    pygame.display.flip()

def draw_icons():
    """
    If in the main world, draw bg sprites and location icons.
    
    If the player is colliding with an icon, return a string of the location name.
    """
    # BG sprites
    for icon_pos in bg_sprite_pos:
        # Convert world position to screen position
        screen_x = icon_pos[0] - camera_offset.x
        screen_y = icon_pos[1] - camera_offset.y

        # Check if the icon is within the screen bounds
        if (0 <= screen_x + BG_SPRITE_SIZE[0] <= base_res[0] and
            0 <= screen_y + BG_SPRITE_SIZE[1] <= base_res[1]):
            # Only draw the icon if it's within the screen view
            game_surface.blit(art_assets['bgnd_sprite'], (screen_x, screen_y))
    
    SHADOW_COLOR = (0, 0, 0, 100)  # Semi-transparent black
    SHADOW_PADDING = 12  # Extra pixels to make the shadow larger

    colliding_loc_icon = None    
    # Location icons
    setup = master_dict['setup']
    for i in range(setup['n_locations']):
        # Get the location name
        loc_name = setup['locations'][i]['name']
        loc_icon = art_assets[loc_name]['icon']
        icon_pos = loc_icon_pos[i]
        
        screen_x = icon_pos[0] - camera_offset.x
        screen_y = icon_pos[1] - camera_offset.y
        
        screen_rect = pygame.Rect(0, 0, base_res[0], base_res[1])
        icon_rect = pygame.Rect(screen_x, screen_y, LOC_ICON_SIZE[0], LOC_ICON_SIZE[1])
        player_screen_x = player_world_pos.x - camera_offset.x
        player_screen_y = player_world_pos.y - camera_offset.y
        player_rect = pygame.Rect(player_screen_x, player_screen_y, PLAYER_SIZE[0], PLAYER_SIZE[1])
        
        # Render if any part of the icon is visible
        if (icon_rect.colliderect(screen_rect)):
            
            # Create a surface with alpha for the shadow
            shadow_surf = pygame.Surface(
                (LOC_ICON_SIZE[0] + SHADOW_PADDING, LOC_ICON_SIZE[1] + SHADOW_PADDING),
                pygame.SRCALPHA
            )
            shadow_surf.fill(SHADOW_COLOR)

            # Center shadow behind the icon
            shadow_x = screen_x - SHADOW_PADDING // 2
            shadow_y = screen_y - SHADOW_PADDING // 2
            game_surface.blit(shadow_surf, (shadow_x, shadow_y))

            # Draw the actual icon on top
            game_surface.blit(loc_icon, (screen_x, screen_y))
            
            # Check if the player is colliding with this one
            if(player_rect.colliderect(icon_rect)): colliding_loc_icon = loc_name
            
    # Render the home icon (art_assets['home_icon']) at position (1536, 960). Size is LOC_ICON_SIZE
    home_rect = pygame.Rect(1536 - camera_offset.x, 960 - camera_offset.y, LOC_ICON_SIZE[0], LOC_ICON_SIZE[1])
    if(home_rect.colliderect(screen_rect)):
        shadow_surf = pygame.Surface(
            (LOC_ICON_SIZE[0] + SHADOW_PADDING, LOC_ICON_SIZE[1] + SHADOW_PADDING),
            pygame.SRCALPHA
        )
        shadow_surf.fill(SHADOW_COLOR)

        # Center shadow behind the icon
        shadow_x = home_rect.x - SHADOW_PADDING // 2
        shadow_y = home_rect.y - SHADOW_PADDING // 2
        game_surface.blit(shadow_surf, (shadow_x, shadow_y))

        # Draw the actual icon on top
        game_surface.blit(art_assets['home_icon'], (home_rect.x, home_rect.y))
    
    return colliding_loc_icon

def draw_popup_text(surface, text:str, position, width=200, height=60):
    """
    Draw a popup text box at the given position.
    text is a string. Font is automatic. This function handles wrapping.
    """
    
    # Create a semi-transparent surface
    popup_surf = pygame.Surface((width, height), pygame.SRCALPHA)
    popup_surf.fill((0, 0, 0, 180))  # Semi-transparent black background

    # Draw a white border
    pygame.draw.rect(popup_surf, (255, 255, 255), popup_surf.get_rect(), 2)

    # Render each line of text
    font = pygame.font.SysFont(None, 24)
    
    text_lines = wrap_text(text, font, width - 20)
    
    line_height = font.get_height()
    total_text_height = line_height * len(text_lines)
    start_y = (height - total_text_height) // 2  # Center vertically

    for i, line in enumerate(text_lines):
        text_surf = font.render(line, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=(width // 2, start_y + i * line_height + line_height // 2))
        popup_surf.blit(text_surf, text_rect)

    # Blit popup to game surface
    surface.blit(popup_surf, position)
 
def draw_location_and_npc(loc_art:pygame.Surface, npc_art:pygame.Surface):
    """
    Draw the current location, and draw the NPC that's there
    """ 
    
    loc_art_scaled = pygame.transform.scale(loc_art, base_res)
    game_surface.blit(loc_art_scaled, (0, 0))
    
    npc_x = int((base_res[0] - NPC_SIZE_SCALED[0]) * 3/4) + 50
    npc_y = base_res[1] - NPC_SIZE_SCALED[1] - 40
    
    npc_art_scaled = pygame.transform.scale(npc_art, NPC_SIZE_SCALED)
    
    game_surface.blit(npc_art_scaled, (npc_x, npc_y))
    
    
def check_enter_pressed(events):
    """
    Return true is the enter key has been pressed, else false.
    
    """
    for event in events:
       if event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
           return True
    return False
 
def handle_text_input(events, surface, in_txt, font, position, prompt_text, width=300, height=100):
    """
    Draw a styled input popup and return updated player input and whether input is finished.
    in_txt is a string, this function automatically handles wrapping.
    """
    # Create a semi-transparent surface
    popup_surf = pygame.Surface((width, height), pygame.SRCALPHA)
    popup_surf.fill((0, 0, 0, 180))  # Semi-transparent black

    # Draw a white border
    pygame.draw.rect(popup_surf, (255, 255, 255), popup_surf.get_rect(), 2)

    # Draw prompt text
    # prompt_text = "Enter your name:"
    prompt_surface = font.render(prompt_text, True, (255, 255, 255))
    prompt_rect = prompt_surface.get_rect(center=(width // 2, 25))
    popup_surf.blit(prompt_surface, prompt_rect)

    # Wrap and draw current input
    wrapped_lines = wrap_text(in_txt, font, width - 20)
    line_height = font.get_height()
    start_y = 50  # Start below prompt

    for i, line in enumerate(wrapped_lines):
        line_surf = font.render(line, True, (255, 255, 255))
        line_rect = line_surf.get_rect(midtop=(width // 2, start_y + i * line_height))
        popup_surf.blit(line_surf, line_rect)

    # Blit popup to screen
    surface.blit(popup_surf, position)

    # Handle input
    for event in events:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                return in_txt, True
            elif event.key == pygame.K_BACKSPACE:
                in_txt = in_txt[:-1]
            elif event.unicode.isprintable():
                in_txt += event.unicode

    return in_txt, False


def wrap_text(text, font, max_width):
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = current_line + word + " "
        if font.size(test_line)[0] <= max_width:
            current_line = test_line
        else:
            lines.append(current_line.strip())
            current_line = word + " "
    if current_line:
        lines.append(current_line.strip())
    return lines

def print_game_info(master_dict, events):
    context_text = master_dict['setup']['context']
    font = pygame.font.Font(None, 36)
    text_color = pygame.Color('white')
    max_width = base_res[0] - 40
    line_spacing = 10
    margin_top = 40
    margin_left = 20

    # Word wrap the context text
    wrapped_lines = [f"Welcome, {master_dict['setup']['name']}. We have been expecting you."]
    wrapped_lines.append("")
    for paragraph in context_text.split('\n'):
        wrapped_lines.extend(wrap_text(paragraph, font, max_width))
        wrapped_lines.append("")  # blank line between paragraphs
            
    wrapped_lines.append("Press enter to continue...")

    # Draw background for readability
    game_surface.fill((0, 0, 0))  # Optional: black background

    # Render text lines
    y = margin_top
    for line in wrapped_lines:
        rendered_line = font.render(line, True, text_color)
        game_surface.blit(rendered_line, (margin_left, y))
        y += rendered_line.get_height() + line_spacing

    # Wait for Enter to continue
    if check_enter_pressed(events):
        return S_MAIN_WORLD

    else: return S_PRINT_GAME_INFO

def main_game_loop():
    running = True
    
    start_time = time.time()
    
    elapsed_time = 0.0
    prev_time = elapsed_time
    
    render_state = S_GET_PLAYER_NAME
    interaction_state = S_NPC_PROMPT_LLM
    current_location = 'main'
    
    while running:
        dt = clock.tick(60)
        dt_s = dt / 1000.0
        elapsed_time += dt_s

        # events = handle_events()
        events = pygame.event.get()
        running = handle_events(events, render_state)
        
        if(render_state == S_MAIN_WORLD):   
            handle_movement()
            clamp_player_to_world()
            scroll_camera_to_player()

            draw_background()
            colliding_icon = draw_icons()
            draw_player()
            draw_minimap()            
        
            # Check if the player is over any icons, and if so, pop up some text
            if(colliding_icon):
                width = 350
                popup_text = f"Press Enter to go into {colliding_icon}"
                popup_position = ((base_res[0] // 2) - width/2, base_res[1] - (base_res[1] // 4))  # Centered bottom
                draw_popup_text(game_surface, popup_text, popup_position, width=width, height=60)
                
                # Check if the enter key was pressed
                if(check_enter_pressed(events)):
                    master_dict['state']['location'] = colliding_icon
                    render_state = S_IN_LOCATION
        
        elif(render_state == S_IN_LOCATION):
            loc = master_dict['state']['location']
            loc_art = art_assets[loc]['full']
            
            npc_name = locations[loc]['npc_name']
            npc_art = art_assets[npc_name]
            draw_location_and_npc(loc_art, npc_art)
            scaled_size = (98, 196)
            draw_player((int((base_res[0] - scaled_size[0]) * 1/4 - 150), base_res[1] - scaled_size[1] - 40), scaled_size)
            
            # Prompt the LLM
            # TODO: make this asynchronous
            # At location <loc> with 
            if(interaction_state == S_NPC_PROMPT_LLM):
                # Render the frame first because prompting the LLM takes a while
                render_frame()
                interaction_state = S_NPC_TALK
                instruction_base = game.get_playtime_prompt(master_dict['setup'], master_dict['state'], master_dict['history'])
                instruction = f"The player is at {loc} during round {master_dict['state']['round']}. Give the player the next logical thing "
                instruction += f"that {npc_name} would say given the current progression of the game. The player's name is {master_dict['setup']['name']}."
                instruction += " Simply continue off from where the conversation last ended - DO NOT INCLUDE ANY OTHER TEXT THAN WHAT THE NPC WOULD SAY. "
                prompt = master_dict['llm_premise'] + instruction_base + instruction
                
                npc_response = prompt_llm(prompt)
                master_dict['history'].append(npc_response)
                write_to_log(f"INFO [frontend.py::main_game_loop()] NPC response: {npc_response}")
                interaction_state = S_NPC_TALK
            
            elif(interaction_state == S_NPC_TALK):
                # Draw popup text of the NPC response at the top-middle of the screen
                width = 680
                popup_text = f"{npc_name}: {npc_response}"
                draw_popup_text(game_surface, popup_text, (base_res[0] // 2 - width/2, 10), width=width, height=120)
                
                # Get the player's input from just below that
                input_box_pos = (base_res[0] // 2 - 150, base_res[1] - 200)  # Centered
                master_dict['player_response'], finished = handle_text_input(events, game_surface, master_dict['player_response'], font, input_box_pos, prompt_text="Your response (or 'LEAVE'): ")
                if finished:
                    # Check if the player response is terminal (go back to main world) or a response to NPC
                    if(master_dict['player_response'] == 'LEAVE'):
                        master_dict['state']['location'] = 'home'
                        master_dict['state']['round'] += 1
                        render_state = S_MAIN_WORLD
                    else:
                        master_dict['history'].append(master_dict['player_response'])
                        master_dict['player_response'] = ""
                    master_dict['player_response'] = ""
                    interaction_state = S_NPC_PROMPT_LLM
            
        elif(render_state == S_GET_PLAYER_NAME):
            font = pygame.font.Font(None, 24)
            input_box_pos = (base_res[0] // 2 - 150, base_res[1] // 2 - 50)  # Centered
            master_dict['setup']['name'], finished = handle_text_input(events, game_surface, master_dict['setup']['name'], font, input_box_pos, prompt_text="Enter your name: ")
            if finished:
                master_dict['setup']['name'] = master_dict['setup']['name'].capitalize()
                render_state = S_PRINT_GAME_INFO  # or your next state
        
        elif(render_state == S_PRINT_GAME_INFO):
            render_state = print_game_info(master_dict, events)

        # Always call this last
        render_frame()
        
        if(elapsed_time > (1.0 + prev_time)):
            # print("Elapsed another second")
            prev_time = elapsed_time
            
    actual_time = time.time() - start_time
    print(f"Elapsed time (actual): {actual_time}")
    print(f"dt calculated time: {elapsed_time}")
    pygame.quit()
    

if __name__ == "__main__":
    
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("CS552 Final Project")
    game_surface = pygame.Surface(base_res)  # low-res render surface
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    bigfont = pygame.font.Font(None, 48)
    
    # Initialize everything else
    # tone = input("Enter the tone of the game (e.g. 'anime', 'fantasy', 'sci-fi'): ")
    # premise = game.generate_initial_premise(desired_tone=tone)
    # game_setup, locations, npcs = game.get_game_setup(premise)
    # art_assets, art_paths = get_game_art(game_setup, locations, npcs)
    # print(json.dumps(art_paths, indent=4))
    
    # Save
    # with open('resources/game_setup_anime.pkl', 'wb') as f:
    #     p = (game_setup, locations, npcs, art_paths, premise)
    #     pickle.dump(p, f)
    
    # Load
    with open('resources/game_setup_anime.pkl', 'rb') as f:
        p = pickle.load(f)
        game_setup, locations, npcs, art_paths, premise = p
        art_assets = load_assets_from_paths(art_paths)
        
    game_state = {
        'round' : 0,
        'location' : 'home'
    }
    game_setup['name'] = ""
    master_dict = {
        'llm_premise':premise,      # Premise of what the LLM is doing
        'setup':game_setup,         # context of the game for the LLM
        'state': game_state,        # 'round' and 'location'
        'history': [],              # List of previous interactions
        'npcs': npcs,               # Dict of NPCs
        'locations': locations,     # Dict of locations
        'player_response':""
    }
            

    # --- Player Setup (World Coordinates) ---
    # Start halfway down in the bottom middle world chunk
    player_world_pos = pygame.Vector2(base_res[0] + base_res[0]//2, base_res[1] + base_res[1]//2)
    camera_offset = pygame.Vector2(0, 0)

    # Define scroll margin: player stays inside this box before scrolling starts
    scroll_margin = pygame.Rect(
        base_res[0] * 0.3,  # left
        base_res[1] * 0.3, # top
        base_res[0] * 0.4 - PLAYER_SIZE[0],  # width
        base_res[1] * 0.4 - PLAYER_SIZE[1] # height
    )

    initialize_icons()
    main_game_loop()        


# from concurrent.futures import ThreadPoolExecutor, as_completed
# import pygame

# def process_location_assets(loc_name, loc_desc, loc_icon_desc, npc_name, npc_desc):
#     try:
#         npc_path, _ = get_pixel_art(npc_desc, size=NPC_SIZE, style="portrait", b_tile=False, rm_bg=True)
#         loc_full_path, loc_b64 = get_pixel_art(loc_desc, size=LOC_SIZE, style="texture", b_tile=False, rm_bg=False)
#         loc_icon_path, _ = get_pixel_art(loc_icon_desc, size=LOC_ICON_SIZE, style="game_asset", b_tile=False, rm_bg=False, input_b64=loc_b64)

#         tiles = {
#             loc_name: {
#                 'full': pygame.image.load(loc_full_path).convert_alpha(),
#                 'icon': pygame.image.load(loc_icon_path).convert_alpha()
#             },
#             npc_name: pygame.image.load(npc_path).convert_alpha()
#         }
#         paths = {
#             loc_name: {
#                 'full_path': loc_full_path,
#                 'icon_path': loc_icon_path
#             },
#             npc_name: npc_path
#         }

#         return {'tiles': tiles, 'paths': paths}
#     except Exception as e:
#         write_to_log(f"ERROR [process_location_assets()] Failed to process {loc_name}: {e}")
#         return {'tiles': {}, 'paths': {}}


# def get_game_art(game_setup:dict, locs:dict, npcs:dict):
#     all_tiles = {}
#     all_paths = {}

#     futures = []
#     with ThreadPoolExecutor(max_workers=10) as executor:
#         # Submit all location-related tasks
#         for loc_name in locs:
#             loc_desc = locs[loc_name]['pixelart_prompt']
#             loc_icon_desc = locs[loc_name]['pixelart_icon']
#             npc_name = locs[loc_name]['npc_name']
#             npc_desc = npcs[npc_name]['npc_art_prompt']

#             write_to_log(f"INFO [frontend.py::get_game_art()] Submitting image tasks for location '{loc_name}'")

#             futures.append(executor.submit(process_location_assets, loc_name, loc_desc, loc_icon_desc, npc_name, npc_desc))

#         # Submit other standalone tasks
#         theme = game_setup['theme']
#         futures.append(executor.submit(get_pixel_art, f"A very lightly-detailed background floor tile with the theme {theme}. The color palette is light.", (TILE_SIZE, TILE_SIZE), "texture", True, False))
#         futures.append(executor.submit(get_pixel_art, f"A Wall tile according to the theme {theme}. This is a world boundary for the game, and should be dense.", (TILE_SIZE, TILE_SIZE), "texture", True, False))
#         futures.append(executor.submit(get_pixel_art, game_setup['special_sprite'], BG_SPRITE_SIZE, "game_asset", False, True))
#         futures.append(executor.submit(get_pixel_art, game_setup['player_art_prompt'], PLAYER_SIZE, "portrait", False, True))

#         # Process results
#         for future in as_completed(futures):
#             result = future.result()
#             if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], str):
#                 # It's a generic image result
#                 path, label = result
#                 surface = pygame.image.load(path).convert_alpha()
#                 all_tiles[label] = surface
#                 all_paths[label] = path
#             elif isinstance(result, dict):
#                 # Location asset dict
#                 all_tiles.update(result['tiles'])
#                 all_paths.update(result['paths'])

#     return all_tiles, all_paths