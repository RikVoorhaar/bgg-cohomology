from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

extensions = [Extension("cohomology", ["cohomology.pyx"])]

setup(name="cohomology", ext_modules=cythonize(extensions))
