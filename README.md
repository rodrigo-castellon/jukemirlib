# jukemirlib

jukemirlib is a Python module that allows you to extract representations from Jukebox in a couple of lines of code.

```Python
import jukemirlib

# this takes a while the first time!
vqvae, top_prior = jukemirlib.setup_models(device='cuda')

audio = jukemirlib.load_audio(fpath)

reps = jukemirlib.extract(vqvae, top_prior, audio, layers=[36])
```

### Installing
You can install via `pip`
```sh
pip install jukemirlib
```
