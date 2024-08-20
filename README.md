# Phase Diversity Phase Retrieval for Microscope Image Reconstruction

## Description
The goal of this project is to provide a number of tools for phase aberration
reconstruction using a set of phase diversity images with Poisson-distributed
noise.

The script dm_calibration.py is intended for the calibration of a
deformable mirror, which can then be used both to introduce phase diversity
for reconstruction, as well as to correct for the the phase aberrations that
are calculated.

The file src/reconstruction.py contains a class for the phase aberration
reconstruction, and can also be run independently as a script.

There are also a number of small utilities for functions such as cropping
images and sets of images.

## Installation

To download the project, clone this repository with
```
git clone https://github.com/husicj/poisson-reconstruction.git
```
