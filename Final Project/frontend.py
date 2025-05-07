import sys
import math
import time
import pygame
from api_calls import *

from game import *

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
PLAYER_SIZE = (64, 128)
player_speed = 5

WORLD_SIZE_TILES = (48, 20)
WORLD_SIZE_PIX = (WORLD_SIZE_TILES[0] * TILE_SIZE, WORLD_SIZE_TILES[1] * TILE_SIZE)
base_res = (TILE_SIZE*16, TILE_SIZE*10) # make it a multiple of tile size

LOC_SIZE = (base_res[0] // 2, base_res[1] // 2)
LOC_ICON_SIZE = (int(TILE_SIZE * 3), int(TILE_SIZE * 3))

zoom = 1
window_size = (base_res[0] * zoom, base_res[1] * zoom)

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
    
    # Save all tiles
    all_tiles['player'] = pygame.image.load(player_path).convert_alpha()
    all_tiles['wall_tile'] = pygame.image.load(wall_tile_path).convert_alpha()
    all_tiles['floor_tile'] = pygame.image.load(floor_tile_path).convert_alpha()
    all_tiles['bgnd_sprite'] = pygame.image.load(bgnd_sprite_path).convert_alpha()
    
    all_paths['player'] = player_path
    all_paths['wall_tile'] = wall_tile_path
    all_paths['floor_tile'] = floor_tile_path
    all_paths['bgnd_sprite'] = bgnd_sprite_path
    
    return all_tiles, all_paths

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
                art_assets[key][subkey] = pygame.image.load(art_paths[key][subkey]).convert_alpha()
        else:
            art_assets[key] = pygame.image.load(art_paths[key]).convert_alpha()
    
    return art_assets
    



if __name__ == "__main__":
    
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("CS552 Final Project")
    game_surface = pygame.Surface(base_res)  # low-res render surface
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    
    # Initialize everything else
    # premise, tone = generate_initial_premise()
    # game_setup, locations, npcs = get_game_setup(premise)
    # art_assets, art_paths = get_game_art(game_setup, locations, npcs)
    # print(json.dumps(art_paths, indent=4))
    
    # Save
    # with open('resources/game_setup_anime.pkl', 'wb') as f:
    #     p = (game_setup, locations, npcs, art_paths)
    #     pickle.dump(p, f)
    
    # Load
    with open('resources/game_setup_anime.pkl', 'rb') as f:
        p = pickle.load(f)
        game_setup, locations, npcs, art_paths = p
        art_assets = load_assets_from_paths(art_paths)
        

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

    running = True
    while running:
        dt = clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Movement ---
        keys = pygame.key.get_pressed()
        if(keys[pygame.K_ESCAPE] or keys[pygame.K_q]):
            pygame.quit()
            sys.exit()
        move = pygame.Vector2(0, 0)
        if keys[pygame.K_w]: move.y -= player_speed
        if keys[pygame.K_s]: move.y += player_speed
        if keys[pygame.K_a]: move.x -= player_speed
        if keys[pygame.K_d]: move.x += player_speed
        player_world_pos += move
        
        # Clamp player to the world
        player_world_pos.x = max(0, min(player_world_pos.x, WORLD_SIZE_PIX[0] - PLAYER_SIZE[0]))
        player_world_pos.y = max(0, min(player_world_pos.y, WORLD_SIZE_PIX[1] - PLAYER_SIZE[1]))

        # --- Determine screen position of player ---
        player_screen_x = player_world_pos.x - camera_offset.x
        player_screen_y = player_world_pos.y - camera_offset.y

        # --- Scroll the camera if player leaves scroll margin ---
        if player_screen_x < scroll_margin.left:
            camera_offset.x -= scroll_margin.left - player_screen_x
        elif player_screen_x > scroll_margin.right:
            camera_offset.x += player_screen_x - scroll_margin.right

        if player_screen_y < scroll_margin.top:
            camera_offset.y -= scroll_margin.top - player_screen_y
        elif player_screen_y > scroll_margin.bottom:
            camera_offset.y += player_screen_y - scroll_margin.bottom

        # --- Draw ---
        game_area_rect = game_surface.get_rect()  # Get screen bounds

        rows = math.ceil(base_res[1] / TILE_SIZE) + 2
        cols = math.ceil(base_res[0] / TILE_SIZE) + 2
        start_x = int(camera_offset.x // TILE_SIZE)
        start_y = int(camera_offset.y // TILE_SIZE)

        # Draw background tiles
        for row in range(start_y, start_y + rows):
            for col in range(start_x, start_x + cols):
                world_x = col * TILE_SIZE
                world_y = row * TILE_SIZE
                screen_x = world_x - camera_offset.x
                screen_y = world_y - camera_offset.y

                rect = pygame.Rect(screen_x, screen_y, TILE_SIZE, TILE_SIZE)
                
                if((0 <= col < WORLD_SIZE_TILES[0]) and (0 <= row < WORLD_SIZE_TILES[1])):
                    game_surface.blit(art_assets['floor_tile'], (screen_x, screen_y))
                else:
                    game_surface.blit(art_assets['wall_tile'], (screen_x, screen_y))
                    
                
                # Debug: draw tile borders
                # pygame.draw.rect(game_surface, (0, 0, 0), rect, 1)  # tile border
                
        # The world is 3x2.

        # Draw player (at screen position)
        player_rect = pygame.Rect(
            player_world_pos.x - camera_offset.x,
            player_world_pos.y - camera_offset.y,
            PLAYER_SIZE[0], PLAYER_SIZE[1]
        )
        game_surface.blit(art_assets['player'], player_rect)
        
        # Draw minimap
        minimap_rect = pygame.Rect(MINIMAP_POS[0], MINIMAP_POS[1], MINIMAP_WIDTH, MINIMAP_HEIGHT)
        pygame.draw.rect(game_surface, (100, 100, 100, 100), minimap_rect)  # Dark grey background
        scale_x = MINIMAP_WIDTH / WORLD_SIZE_PIX[0]
        scale_y = MINIMAP_HEIGHT / WORLD_SIZE_PIX[1]

        minimap_player_x = int(MINIMAP_POS[0] + player_world_pos.x * scale_x)
        minimap_player_y = int(MINIMAP_POS[1] + player_world_pos.y * scale_y)

        pygame.draw.circle(game_surface, (255, 0, 0), (minimap_player_x, minimap_player_y), 2)  # Red dot for player

        camera_rect = pygame.Rect(
            MINIMAP_POS[0] + camera_offset.x * scale_x,
            MINIMAP_POS[1] + camera_offset.y * scale_y,
            base_res[0] * scale_x,
            base_res[1] * scale_y
        )
        pygame.draw.rect(game_surface, (255, 255, 255), camera_rect, 1)  # White outline  

        # Draw scroll margin box for debugging
        # pygame.draw.rect(game_surface, (255, 255, 255), scroll_margin, 2)

        # Transform the surface to the screen
        scaled_surface = pygame.transform.scale(game_surface, window_size)
        screen.blit(scaled_surface, (0, 0))
        
        pygame.display.flip()

    pygame.quit()
    sys.exit()
