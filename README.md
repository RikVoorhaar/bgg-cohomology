## bgg-cohomology

This is an implementation of a computer algorithm to compute the BGG cohomology of a Lie algebra module. 

## Usage

This module requires `sagemath`, and currently also `numpy-indexed`. 
To install this module, clone or download the repository. Then use `import` statements in sage to 
use the package. 

Examples of usage of this package are in the in the form of Jupyter notebooks in the `examples` directory.

## To do

Several important features are currently in development.
- Use `LinBox` for the linear algebra computations to improve performance.  
- Parallelize the computationally intensive parts of the algorithm
- Add support for a wider class of Lie algebra modules. In particular quotient modules.

If you require any new features for research purposes, please do not hesitate to ask us.


## Credit
All the code has been written by **Rik Voorhaar**. 
 Additional credit goes to **Nicolas Hemelsoet** for many aspects of the algorithm implemented here, 
 and **Anna Lachowska** who gave a lot of advice on the mathematical aspects of this project. 
 This work has been partially financially supported by **NCCR SwissMAP**. 
 
 When using this program for academic purposes, please cite the relevant paper on the arXiv. 
 (Link to be added once the paper is actually on the arXiv.)