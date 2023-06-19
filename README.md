# Python package for building and solving Sparse Layered Graphs
This package allows building and solving image segmentation problems such as [Markov Random Fields](https://en.wikipedia.org/wiki/Markov_random_field) (MRF) and [Sparse Layered Graphs](http://openaccess.thecvf.com/content_CVPR_2020/papers/Jeppesen_Sparse_Layered_Graphs_for_Multi-Object_Segmentation_CVPR_2020_paper.pdf) (SLG) using *s-t* graph cuts. The package itself is written purely in Python and contains logic for building graphs for both single- and multi-label problems. To solve the optimization problem it relies on a min-cut/max-flow algorithm (see [Solvers](#solvers)).

## Installation
Install default package using `pip install slgbuilder` or clone the repository. See [Dependencies](#dependencies) for more.

## What is it for?
The package is primarily targeted multi-label/multi-object image segmentation problems. Common uses include:
- Surface detection
- Constrained multi-surface detection
- Object segmentation
- Interacting multi-object segmentation

### CVPR 2020 teaser video
[![Teaser video](https://img.youtube.com/vi/CFUYuL1J85k/0.jpg)](https://www.youtube.com/watch?v=CFUYuL1J85k)

### Examples
- [Example code and notebooks](https://github.com/Skielex/slgbuilder-examples)
- [Experimental notebooks (advanced 2D and 3D segmentation) [CVPR2020]](https://doi.org/10.11583/DTU.12016941)
- [Experimental notebooks (advanced 3D segmentation with parallel QPBO) [ICCV 2021]](https://doi.org/10.5281/zenodo.5201619)

## Grid-graph vs. ordered multi-column graph
The package support both the common grid-graph structure, as used by [Delong and Boykov](https://doi.org/10.1109/ICCV.2009.5459263) and the ordered multi-column structure, popularized by [Li et al.](https://doi.org/10.1109/TPAMI.2006.19). Although representing image data in the grid structure may seem like the obvious choice, there are several advantages (e.g. smaller graph and "better" constraints) when using the ordered multi-column structure for segmentation if possible. However, doing so requires resample the data, which usually requires knowledge about the location of each object in the image.

## Solvers
The package currently supports eight different solvers. In simple cases the default BK Maxflow and QPBO solvers should do fine. For submodular problems, all solver should give find a globally optimal solution. Only the QPBO solvers support nonsubmodular problems, and for such problems a complete solution cannot be guaranteed. For large or time-critical tasks it may be favorable to use a non-default solver, e.g., HPF or PQPBO. We are currently working on a comparative study of min-cut/max-flow algorithms, which should provide better insights into which algorithms perform the best on different computer vision tasks. If you're interested, check out this [repository](https://github.com/patmjen/maxflow_algorithms).

### BK Maxflow
For submodular problems (e.g. segmentation without exclusion constraints), the default solver, used by the `MaxflowBuilder` is an [updated version](https://github.com/Skielex/maxflow) the Boykov-Kolmogorov Maxflow algorithm, accessed through the [`thinmaxflow`](https://github.com/Skielex/thinmaxflow) package. `MaxflowBuilder` is synonymous with `BKBuilder`.

### QPBO
For nonsubmodular problems (e.g. segmentation with exclusion constraints) the solver used by the `QPBOBuilder` is [this version](https://github.com/Skielex/maxflow) of the QPBO algorithm, accessed through the [`thinqpbo`](https://github.com/Skielex/thinqpbo) package.

### HPF
The `HPFBuilder` is an alternative to the standard `MaxflowBuilder`. It relies on the HPF algorithm, which is often superior in performance compared to BK Maxflow. `HPFBuilder` depends on the [`thinhpf`](https://github.com/Skielex/thinhpf) package. It supports `int32` and `float32` capacities/energies, however, `int32` is recommended.

### OR Tools
An alternative to the BK Maxflow solver is the [Google Maxflow](https://developers.google.com/optimization/flow/maxflow) implementation, which is a push-relabel algorithm. This can be done using the `ORBuilder` class. Apart from performance, the difference between the Google and BK Maxflow algorithms is that the Google implementation doesn't support floating type capacities. If `MaxflowBuilder` is slow when solving, try using the `ORBuilder` instead.

### MBK
A slightly optimized modern reimplementation of the BK Maxflow algorithm is available using the `MBKBuilder`. Depends on the [`shrdr`](https://github.com/Skielex/shrdr) package. Using this builder should reduce memory usage and increase performance slightly compared to the `MaxflowBuilder`/`BKBuilder`.

### PBK
A parallel version of the `MBK` implementation found in the [`shrdr`](https://github.com/Skielex/shrdr) package. Can significantly reduce solve for large problems on multicore systems. Jupyter notebooks with examples using the similar `PQPBOBuilder` can be found in the [supplementary material](https://doi.org/10.5281/zenodo.5201619) of this [article](https://openaccess.thecvf.com/content/ICCV2021/papers/Jeppesen_Faster_Multi-Object_Segmentation_Using_Parallel_Quadratic_Pseudo-Boolean_Optimization_ICCV_2021_paper.pdf).

### MQPBO
A slightly optimized modern reimplementation of the QPBO algorithm is available using the `QPBOBuilder`. Depends on the [`shrdr`](https://github.com/Skielex/shrdr) package. Using this builder should reduce memory usage and increase performance slightly compared to the `QPBOBuilder`.

### PQPBO
A parallel version of the `MQPBO` implementation found in the [`shrdr`](https://github.com/Skielex/shrdr) package. Can significantly reduce solve for large problems on multicore systems. Jupyter notebooks with examples of use can be found in the [supplementary material](https://doi.org/10.5281/zenodo.5201619) of this [article](https://openaccess.thecvf.com/content/ICCV2021/papers/Jeppesen_Faster_Multi-Object_Segmentation_Using_Parallel_Quadratic_Pseudo-Boolean_Optimization_ICCV_2021_paper.pdf).


## Dependencies
 To solve the *s-t* graph-cut optimization problems the package relies on one of the following packages:
- [`thinmaxflow`](https://github.com/Skielex/thinmaxflow) (installed by default) - Package for computing min-cut/max-flow of an *s-t* graph using the Boykov-Kolmogorov (BK) Maxflow algorithm.
- [`thinqpbo`](https://github.com/Skielex/thinqpbo) (installed by default) - Package for solving submodular and nonsubmodular optimization problems using the Quadratic Pseudo-Boolean Optimization (QPBO) algorithm implementation by Kolmogorov. Solver based on the BK Maxflow.
- [`thinhpf`](https://github.com/Skielex/thinhpf)- Package for computing min-cut/max-flow of an *s-t* graph using the Hochbaum Pseudoflow (HPF) algorithm.
- [`ortools`](https://github.com/google/or-tools) - Package for computing min-cut/max-flow of an *s-t* graph using the Google OR Tools min-cut/max-flow implementation.
- [`shrdr`](https://github.com/Skielex/shrdr) - Package for computing min-cut/max-flow of an *s-t* graph optimized serial or parallel implementations of BK Maxflow and QPBO algorithms by Jensen and Jeppesen.

See links for further details and references.

### Install with extra dependencies
Install with HPF solver.
```
pip install slgbuilder[thinhpf]
```
Install with Google OR Tools solver.
```
pip install slgbuilder[ortools]
```
Install with all extras.
```
pip install slgbuilder[all]
```
The `shrdr` package is not yet available on PyPI. Get it [here](https://github.com/Skielex/shrdr)!

## Related repositories
- [shrdr](https://github.com/Skielex/shrdr) Python package (ICCV 2021)
- [thinqpbo](https://github.com/Skielex/thinqpbo) Python package
- [thinmaxflow](https://github.com/Skielex/thinmaxflow) Python package
- [thinhpf](https://github.com/Skielex/thinhpf) Python package
- [C++ implementations](https://github.com/patmjen/maxflow_algorithms) of min-cut/max-flow algorithms

## Contributions
Contributions are welcome, just create an [issue](https://github.com/Skielex/slgbuilder/issues) or a [PR](https://github.com/Skielex/slgbuilder/pulls).

## License
MIT License (see LICENSE file).

## Reference
If you use this any of this for academic work, please consider citing our work:
> N. Jeppesen, A. N. Christensen, V. A. Dahl and A. B. Dahl, "Sparse Layered Graphs for Multi-Object Segmentation," 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 12774-12782, doi: 10.1109/CVPR42600.2020.01279.<br>
[ [paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Jeppesen_Sparse_Layered_Graphs_for_Multi-Object_Segmentation_CVPR_2020_paper.pdf) ]
[ [supp](https://doi.org/10.11583/DTU.12016941) ]
[ [CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Jeppesen_Sparse_Layered_Graphs_for_Multi-Object_Segmentation_CVPR_2020_paper.html) ]
[ [IEEE](https://doi.org/10.1109/CVPR42600.2020.01279) ]


> N. Jeppesen, P. M. Jensen, A. N. Christensen, A. B. Dahl and V. A. Dahl, "Faster Multi-Object Segmentation using Parallel Quadratic Pseudo-Boolean Optimization," 2021 IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 6240-6249, doi: 10.1109/ICCV48922.2021.00620.<br>
[ [paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Jeppesen_Faster_Multi-Object_Segmentation_Using_Parallel_Quadratic_Pseudo-Boolean_Optimization_ICCV_2021_paper.pdf) ]
[ [supp](https://doi.org/10.5281/zenodo.5201619) ]
[ [CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Jeppesen_Faster_Multi-Object_Segmentation_Using_Parallel_Quadratic_Pseudo-Boolean_Optimization_ICCV_2021_paper.html) ]
[ [IEEE](https://doi.org/10.1109/ICCV48922.2021.00620) ]


### BibTeX

``` bibtex
@INPROCEEDINGS{9156301,  author={Jeppesen, Niels and Christensen, Anders N. and Dahl, Vedrana A. and Dahl, Anders B.},  booktitle={2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},   title={Sparse Layered Graphs for Multi-Object Segmentation},   year={2020},  volume={},  number={},  pages={12774-12782},  doi={10.1109/CVPR42600.2020.01279}}

@INPROCEEDINGS{9710633,  author={Jeppesen, Niels and Jensen, Patrick M. and Christensen, Anders N. and Dahl, Anders B. and Dahl, Vedrana A.},  booktitle={2021 IEEE/CVF International Conference on Computer Vision (ICCV)},   title={Faster Multi-Object Segmentation using Parallel Quadratic Pseudo-Boolean Optimization},   year={2021},  volume={},  number={},  pages={6240-6249},  doi={10.1109/ICCV48922.2021.00620}}
```