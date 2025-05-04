INIT_GAME_PROMPT = """
Generate an initial game state. I need the following information:
First, The names of five characters, each bound to one of the given locations
(Underground Speakeasy, Atwater Kent Electrical Engineering building, Wizard's Lab, Library, Roman Colosseum)
the character name can be retrieved by accessing json['characters'][location]['name'].
Also include a description (key 'description'), a concise sentence about personality.
This can be retrieved at json['characters'][location]['description']

Second, I also need to know what the passphrase is, and who holds that information (the name of one of the characters).
These keys should be the strings 'passphrase' and 'passphrase_holder' in the top level of the json.
Do not include any quotation marks or apostrophes in the passphrase

Third, set the environment of the game. This will be used to generate a texture map, so make it detailed enough
for a diffusion model to generate from (2-3 sentences). Have this under the key 'evironment'

Lastly, give me a three-sentence piece of context that can be fed back into an LLM to continue running the game. This should be 
under the key 'context'. You can make the goal of the game to be whatever you want, as long as the player needs to find a key.

Format your response as a json string, with character names (as strings in quotation marks)
as the top-level keys in the 'characters' dictionary.

Summary of format:
{
"characters":
    -- "[location name]":
    --  -- "name" : the name of the character
    --  -- "description" : what the NPC is like
"passphrase" : you decide
"passphrase_holder": one of the NPCs
"context" : the context
"environment" : the environment
}
"""

"""
Example json response for first prompt:
{
"characters": {
    "bar": {
        "name": "Mack the Mixologist",
        "description": "Jovial bartender who insists every problem can be shaken or stirred."
    },
    "ak": {
        "name": "Professor Wattson",
        "description": "Eccentric engineer, obsessed with electricity and fond of dramatic hand gestures."
    },
    "lab": {
        "name": "Fizzicus the Wobbly Wizard",
        "description": "Absent-minded, enthusiastic about magical mishaps and explosive experiments."
    },
    "library": {
        "name": "Agnes Bookbinder",
        "description": "Serious, shushing librarian who secretly enjoys bad puns and forbidden lore."
    },
    "rome": {
        "name": "Maximus Decimus Quirkus",
        "description": "Overly dramatic gladiator who quotes ancient proverbs, usually incorrectly."
    }
},
"passphrase": "wombat dance party",
"passphrase_holder": "Fizzicus the Wobbly Wizard",
"evironment": "A surreal campus stitched together by improbable physics: the colossal colosseum looms a stone�s throw from a glimmering laboratory filled with smoke, while the Atwater Kent building crackles with electrical arcs. The ancient library towers with impossible book-laden spires, and underground, a cozy bar radiates golden light and muffled laughter. Above it all, a patchwork sky shifts between thunderclouds, magical auroras, and Roman sunlight.",
"context": "You awaken at home with no memory of how you got there or how to leave. To escape, you must find a secret key phrase tied to one of the five quirky characters scattered around this bizarre campus. Each NPC knows who has the phrase, but some might point you in the wrong direction."
}
"""