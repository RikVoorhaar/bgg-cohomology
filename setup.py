import setuptools
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

setuptools.setup(name="bgg-cohomology",
version='1.6',
autho="Rik Voorhaar",
url="https://github.com/RikVoorhaar/bgg-cohomology",
ext_modules=cythonize("bggcomplex/*.pyx"),
packages=setuptools.find_packages(where='bggcomplex'))
