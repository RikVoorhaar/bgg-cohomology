import setuptools
from Cython.Build import cythonize

setuptools.setup(
    name="bggcohomology",
    version="1.6",
    autho="Rik Voorhaar",
    url="https://github.com/RikVoorhaar/bgg-cohomology",
    ext_modules=cythonize("bgg-cohomology/bggcohomology/*.pyx"),
    packages=["bgg-cohomology"],
    install_requires=["tqdm"],
)
