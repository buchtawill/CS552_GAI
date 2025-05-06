
LOG_NAME = 'gamelog.log'

def write_to_log(s:str, level=99):
    try:
        with open(LOG_NAME, 'a') as f:
            f.write(s+'\n')
            f.flush()
    except Exception as e:
        print(f"ERROR [log.py::write_to_log()] Error writing to log: {e}")