# jukemirlib

jukemirlib is a Python module that allows you to extract representations from Jukebox in a couple of lines of code.

```Python
import jukemirlib

audio = jukemirlib.load_audio(fpath)

reps = jukemirlib.extract(audio, layers=[36])
```

The first time you run `jukemirlib.extract()`, it'll take about an hour (depending on your internet speed), since it needs to download the Jukebox weights and cache them. Then, the first time you run `jukemirlib.extract()` within a single Python thread it will take pretty long, since it has to load the whole model into VRAM, but every subsequent run will be quick and easy.

Changing defaults:
- The default model weight cache directory is `~/.cache/jukemirlib`, but if you want to change that, just set `jukemirlib.CACHE_DIR` to your desired directory before continuing with extraction.
- The default device is "cuda" if you have a GPU and "cpu" otherwise, but if you want to set it manually you can set `jukemirlib.DEVICE`.

### Installing
You can install via `pip`
```sh
pip install jukemirlib
```
