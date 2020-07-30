Installation and Usage
======================

Installation
------------

This project relies on Sagemath, for installation instructions
see: `install Sagemath <https://doc.sagemath.org/html/en/installation/>`_.
The most recent version of this package is developed with Sagemath 9.0, but
more recent versions should also work. 

After installing `sage` run the following command::

    sage -pip install git+https://github.com/RikVoorhaar/bgg-cohomology.git

In MacOS and linux this should run just fine. When using Windows, 
make sure to run this command from the Sagemath Shell. 

To open the `.ipynb` files in the `computations` and `examples` folders, you
can either launch `Sagemath Notebook` or run the following in the command line::

    sage -n

Usage
-----
We strongly recommend reading the :doc:`general tutorial </tutorial>` to get started
If you want to compute the BGG cohomology of a lie algebra module defined
as a quotient, see the :doc:`tutorial on using cokernels </cokernels>`. 
Finally for an introduction on how to use the functionality related to
computing the Hochschild cohomology of (partial) flag varieties / the center
off the small quantum group, see the :doc:`quantum center tutorial </quantum_center>`.

Furthermore, the notebooks and scripts in the `computations` folder can
also serve as examples. These files were used to generate the results
published `our arXiv preprint <https://arxiv.org/abs/1911.00871>`_
as well as for a paper yet to be published by the same authors
on the center of the small quantum group. 