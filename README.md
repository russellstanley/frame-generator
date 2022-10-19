# HATS Generation
This is a custom module for the DV software which can generate a Histogram of Average Time Surfaces (HATS). The original paper can be found [here](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sironi_HATS_Histograms_of_CVPR_2018_paper.pdf). This code is essentially a C++ port of the original python implementation found [here](https://github.com/rfma23/HATS).

<img src="images/sample.png?raw=true" alt="Sample HATS Representation"/> </br>

## Build
Follow the instruction on the [DV](https://inivation.gitlab.io/dv/dv-docs/docs/getting-ready-for-development/) website. Module can be build using the following.
```
 cmake .
 make -j2 -s 
 sudo make install
```
