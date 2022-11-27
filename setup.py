from setuptools import setup, find_packages

setup(
    name="jukemirlib",
    packages=["jukemirlib"],
    install_requires=[
        "jukebox @ git+https://github.com/rodrigo-castellon/jukebox.git",
        "wget",
        "accelerate",
    ],
)
