import setuptools
from Cython.Build import cythonize

setuptools.setup(
    name="bgg-cohomology",
    version="1.6",
    autho="Rik Voorhaar",
    url="https://github.com/RikVoorhaar/bgg-cohomology",
    ext_modules=cythonize("bggcohomology/*.pyx"),
    packages=["bggcohomology"],
    install_requires=["tqdm"],
)
