## bgg-cohomology

This is an implementation of a computer algorithm to compute the BGG resolution and its cohomology of a Verma module. 

You can find the documentation for this package right here:
https://bgg-cohomology.readthedocs.io/en/latest/

## Installation

After installing `sage` run the following command:
```
    sage -pip install bggcohomology
```
In MacOS and linux this should run just fine. When using Windows,
make sure to run this command from the Sagemath Shell (or properly add sage to PATH).

To install the latest version directly from GitHub, run:
```
    sage -pip install git+https://github.com/RikVoorhaar/bgg-cohomology.git
```
Note that this version may be less stable.

To open the `.ipynb` files in the `computations` and `examples` folders, you need to
first of all clone this repository. Then you can either launch `Sagemath Notebook` application if installed,
or run the following in the command line:

```
    sage -n
```

## Usage

There are several tutorials of how to use this module in the `examples` directory.
These same tutorials can also be found in the docs. 
Furthermore the `computations` directory contains notebooks and scripts with the code we used
for the results in our preprint, and upcoming preprint.

## Credit
All the code has been written by **Rik Voorhaar**. 
 Additional credit goes to **Nicolas Hemelsoet** for many aspects of the algorithm implemented here.
 This work has been partially financially supported by **NCCR SwissMAP**. 
 
This software is free to use and edit. When using this software for academic purposes, please cite 
the following preprint: https://arxiv.org/abs/1911.00871
