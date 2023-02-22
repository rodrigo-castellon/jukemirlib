import torch as t
import os

global VQVAE, TOP_PRIOR, DEVICE
VQVAE = None
TOP_PRIOR = None
CACHE_DIR = os.path.expanduser("~") + "/.cache/jukemirlib"
DEVICE = "cuda" if t.cuda.is_available() else "cpu"
T = 8192
JUKEBOX_SAMPLE_RATE = 44100
# 1048576 found in original paper, last page
CTX_WINDOW_LENGTH = 1048576

# if on google cloud, this one is better
# REMOTE_PREFIX = "https://storage.googleapis.com/jukebox-weights/"

# for stability, original endpoint is this one
REMOTE_PREFIX = "https://openaipublic.azureedge.net/jukebox/models/5b/"
