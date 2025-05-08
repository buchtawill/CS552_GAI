To run the project, you must have the following defined in the file "api_key.py":
    - OPENAI_API_KEY = "your openai api key"
    - OPENAI_MODEL_SNAPSHOT = "gpt-4.1-2025-04-14" # or another model of your choice
    - RETRO_DIFFUSION_API_KEY = "your retro diffusion API key"

A text-based version of the game can be played by running game.py. The GUI version can be run with frontend.py.

This project was developed using Python 3.12.4. To create the environment, run 
    1. "/path/to/python3.12 -m virtualenv promptquest_env"
    2. Activate promptquest_env
    3. pip install -r requirements.txt


