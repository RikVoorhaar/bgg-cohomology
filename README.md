## bgg-cohomology

This is an implementation of a computer algorithm to compute the BGG resolution and its cohomology 
of a Verma module. 

## Installation

Install sagemath. The code was developed in version 8.6, and has been tested to work with sage 8.9.

Clone or download the repository, and run the following command in the `bggcomplex` folder to 
compile the code before first usage. 
    
    sage setup.py build_ext -i

There is a basic tutorial of how to use this module in the `examples` directory. 
Furthermore the `computations` directory contains notebooks with the code we used
for the results in our preprint. 

## To do

Several important features are currently in development.
- Optimize the computation of BGG maps
- Add support for a wider class of Lie algebra modules.

We welcome all suggestions and feature requests for academic purposes. 

## Credit
All the code has been written by **Rik Voorhaar**. 
 Additional credit goes to **Nicolas Hemelsoet** for many aspects of the algorithm implemented here.
 This work has been partially financially supported by **NCCR SwissMAP**. 
 
This software is free to use and edit. When using this software for academic purposes, please cite 
the following preprint: https://arxiv.org/abs/1911.00871
