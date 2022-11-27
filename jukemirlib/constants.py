import torch as t
import os

global VQVAE, TOP_PRIOR, DEFAULT_CACHE_PATH, DEVICE
VQVAE = None
TOP_PRIOR = None
CACHE_DIR = os.path.expanduser("~") + "/.cache/jukemirlib"
DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'
