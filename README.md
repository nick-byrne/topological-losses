# Topological losses
Topological losses for spatially coherent CNN-based, multi-class image segmentation in 2D and 3D.

Nick Byrne

## Summary and citation

This repository provides code to demonstrate the application of multi-class, persistent homology (PH)-based topological loss functions within a CNN-based image segmentation post-processing framework, reproducing a result described in:

__[N. Byrne, J. R. Clough, I. Valverde, G. Montana and A. P. King, "A persistent homology-based topological loss for CNN-based multi-class segmentation of CMR," in IEEE Transactions on Medical Imaging, 2022, doi: 10.1109/TMI.2022.3203309.](https://doi.org/10.1109/TMI.2022.3203309)__

(also available as an __[arXiv](https://doi.org/10.48550/arXiv.2107.12689)__ pre-print)

The explanation below is informed by this publication.

## Setup

To run the example provided in `topological-post-processing.ipynb`, you will need to firstly install the following Python packages and their dependencies:

- PyTorch (tested with version 1.7.1 and CUDA Toolkit 9.2)
- CubicalRipser (made available by Shizuo Kaji on __[GitHub](https://github.com/shizuo-kaji/CubicalRipser_3dim)__)

## Contents and advice

Our implementation provides a means of integrating the amazing routines provided by CubicalRipser for computing persistent homology (both rapidly and efficiently) with the automatic differentiation engine of PyTorch. This allows for the resulting topological features of 2D and 3D data (once represented by cubical complexes) to be leveraged within gradient-based learning.

To do so, we implement a (very thin!) wrapper around CubicalRipser via the `get_differentiable_barcode` function found within `topo.py`. For those seeking to investigate, apply or extend our approach, we provide an example of how this wrapper can be used to build topological loss functions for CNN test time adaptation via `multi_class_topological_post_processing`. In this example, the wrapper is used as follows:  

1. Establish a PyTorch `tensor` whose topological features are of interest.
2. Convert the `tensor` to a NumPy `array` and compute the persistence `barcode` of topological features using CubicalRipser (where CPU resources allow, optionally using parallel `multiprocessing`).
3. Call `get_differentiable_barcode`, passing the original `tensor` and `barcode` as arguments, to localise topological features and their persistence.
4. Interrogate the differentiable `barcode` to construct a topological loss between predicted and anticipated topology.

We suggest this recipe as a means for applications of our implementation to other tasks.

## Acknowledgement

We are very grateful to Shizuo Kaji for his work in developing CubicalRipser, an amazing resource, and his responsiveness to our queries relating to our own objectives.

## Contact

Please explore our code, and let us know of any faults, queries or improvements.
