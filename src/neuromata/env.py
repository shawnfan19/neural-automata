import os

try:
    AUTOMATON_DATA_DIR = os.environ["AUTOMATON_DATA_DIR"]
    AUTOMATON_CKPT_DIR = os.environ["AUTOMATON_CKPT_DIR"]
except KeyError as e:
    raise EnvironmentError(f"required environment variable(s) not set: {e}")
