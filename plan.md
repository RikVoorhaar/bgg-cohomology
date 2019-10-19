We want to improve the current code. We want to achieve the following things:
- Add support for quantum module and quotient modules to fast_modules
- Use LinBox for matrix computations in compute_maps
- Rewrite compute_maps in Cython
- Parallelize all of the  performance critical code
- Write a module that has the function of numpy-indexed
- Make efficientllow-level implementation of PBW universal enveloping algebra

A lot of these things require Cython, therefore I propose the following road towards achieving this
- Write simple computational code in Cython. Get comfortable with the interface
	+ Make program that eats numpy matrix and outputs its determinant or sum or something
	+ Make implementation of Lie algebra class in Cython
	+ Make simple parallelized code in Cython
	+ Parallelize the above, and vectorize so that we can compute Lie brackets between large arrays of stuff quickly
- Learn how to use LinBox and try to feed it numpy matrices
- make PBW implementation, and make a division algorithm
- rewrite compute_maps to use these
