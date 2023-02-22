# jukemirlib

jukemirlib is a Python module that allows you to extract representations from Jukebox in a couple of lines of code.

```Python
import jukemirlib

audio = jukemirlib.load_audio(fpath)

reps = jukemirlib.extract(audio, layers=[36])
```

The first time you run `jukemirlib.extract()`, it'll take about an hour (depending on your internet speed), since it needs to download the Jukebox weights and cache them. Then, the first time you run `jukemirlib.extract()` within a single Python thread it will take pretty long (close to a minute), since it has to load the whole model into VRAM, but every subsequent run will be quick and easy.

To change default settings, simply run, as an example
```Python
jukemirlib.setup_models(cache_dir="/path/to/custom/cache/dir", remote_prefix="remote.prefix/url/here", device="cuda:3")
```

This will set up the models with your specified arguments rather than the default constants found in `constants.py`. `cache_dir` specifies where the Jukebox model weights will be cached on your disk (`~/.cache/jukemirlib` by default), `remote_prefix` specifies where we are downloading the weights from ("https://openaipublic.azureedge.net/jukebox/models/5b/" by default), and `device` specifies the device to place weights on ("cuda" if CUDA is available and "cpu" otherwise by default).

If you wish to use this library with your GPU, it must have at least 13GB of VRAM.

### Installing
You can install via `pip`
```sh
pip install git+https://github.com/rodrigo-castellon/jukemirlib.git
```
