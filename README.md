# Python package for building and solving Sparse Layered Graphs
This package allows building and solving [Sparse Layered Graphs](http://openaccess.thecvf.com/content_CVPR_2020/papers/Jeppesen_Sparse_Layered_Graphs_for_Multi-Object_Segmentation_CVPR_2020_paper.pdf) (SLG) using *s-t* graph cuts. The package itself is written purely in Python and contains logic for building graphs for both single- and multi-label problems. To perform the graph-cut the package relies on either the [```thinmaxflow```](https://github.com/Skielex/thinmaxflow), [```thinqpbo```](https://github.com/Skielex/thinqpbo) or [```ortools```](https://github.com/google/or-tools) package.

## Installation
Install package using ```pip install slgbuilder``` or clone the repository.

## What is it for?
The package is primarily targeted multi-label/multi-object image segmentation problems. Common uses include:
- Surface detection
- Constrained multi-surface detection
- Object segmentation
- Interacting multi-object segmentation

### Teaser video
[![Teaser video](https://img.youtube.com/vi/CFUYuL1J85k/0.jpg)](https://www.youtube.com/watch?v=CFUYuL1J85k)

### Examples
- [Example code and notebooks](https://github.com/Skielex/slgbuilder-examples)
- [Experimental notebooks (advanced 2D and 3D segmentation)](https://doi.org/10.11583/DTU.12016941)

## Grid-graph vs. ordered multi-column graph
The package support both the common grid-graph structure, as used by [Delong and Boykov](https://doi.org/10.1109/ICCV.2009.5459263) and the ordered multi-column structure, popularized by [Li et al.](https://doi.org/10.1109/TPAMI.2006.19). Although representing image data in the grid structure may seem like the obvious choice, there are several advantages (e.g. smaller graph and "better" constraints) when using the ordered multi-column structure for segmentation if possible. However, doing so requires resample the data, which usually requires knowledge about the location of each object in the image.

## Solvers
The package currently supports three different.

**BK Maxflow**
For submodular problems (e.g. segmentation without exclusion constriants), the default solver, used by the ```MaxflowBuilder``` is an [updated version](https://github.com/Skielex/maxflow) the Boykov-Kolmogorov Maxflow algorithm, accessed through the [```thinmaxflow```](https://github.com/Skielex/thinmaxflow) package.

**QPBO**
For non-submodular problems (e.g. segmentation with exclusion constriants) the solver used by the ```QPBOBuilder``` is [this version](https://github.com/Skielex/maxflow) of the QPBO algorithm, accessed through the [```thinqpbo```](https://github.com/Skielex/thinqpbo) package.

**OR Tools**
An alternative to the BK Maxflow solver is the [Google Maxflow](https://developers.google.com/optimization/flow/maxflow) implementation, which is a push-relabel algorithm. This can be done using the ```ORBuilder``` class. Apart from performance, the difference between the Google and BK Maxflow algorithms is that the Google implementation doesn't support floating type capacities. If ```MaxflowBuilder``` is slow when solving, try using the ```ORBuilder``` instead.

## Contributions
Contributions are welcome, just create an [issue](https://github.com/Skielex/slgbuilder/issues) or a [PR](https://github.com/Skielex/slgbuilder/pulls).

## License
MIT License (see LICENSE file).

## Reference
If you use this for academic work, please consider citing our paper, [Sparse Layered Graphs for Multi-Object Segmentation](http://openaccess.thecvf.com/content_CVPR_2020/papers/Jeppesen_Sparse_Layered_Graphs_for_Multi-Object_Segmentation_CVPR_2020_paper.pdf).
