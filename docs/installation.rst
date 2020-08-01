Installation and Usage
======================

Installation
------------

After installing `sage` run the following command:

    sage -pip install bggcohomology
    
In MacOS and linux this should run just fine. When using Windows,
make sure to run this command from the Sagemath Shell (or properly add sage to PATH).

To install the latest version directly from GitHub, run:

    sage -pip install git+https://github.com/RikVoorhaar/bgg-cohomology.git

Note that this version may be less stable.

To open the `.ipynb` files in the `computations` and `examples` folders, you need to
first of all clone this repository. Then you can either launch `Sagemath Notebook` application if installed,
or run the following in the command line:

    sage -n

Usage
-----
We strongly recommend reading the :doc:`general tutorial </tutorial>` to get started.
If you are only interested in the BGG complex itself, then look at the 
:doc:`maps in BGG complex tutorial </maps_nb>`.
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