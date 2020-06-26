## bgg-cohomology

This is an implementation of a computer algorithm to compute the BGG resolution and its cohomology 
of a Verma module. 

## Installation

Install sagemath. The code is currently developed for sage 9.0, but should still work on 8.9.

Download the latest version from the `releases`tab. Run the following command in the `bggcomplex` folder to 
compile the code before first usage. 
    
    sage setup.py build_ext -i

Some parts require `tdqm` to display progress bars. This can be installed by running

    sage -pip install tqdm
    
## Usage

There is a basic tutorial of how to use this module in the `examples` directory. 
Furthermore the `computations` directory contains notebooks with the code we used
for the results in our preprint. 

## Credit
All the code has been written by **Rik Voorhaar**. 
 Additional credit goes to **Nicolas Hemelsoet** for many aspects of the algorithm implemented here.
 This work has been partially financially supported by **NCCR SwissMAP**. 
 
This software is free to use and edit. When using this software for academic purposes, please cite 
the following preprint: https://arxiv.org/abs/1911.00871
