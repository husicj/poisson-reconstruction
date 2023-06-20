# Deformable Mirror Calibration (v0.1.1)

## Description

The goal of this project is to provide a streamlined process for calibration of a deformable mirror. Specifically, making use of a phase diversity method and a algorithm based on the assumption of Poisson-distributed noise, the wavefront aberrations introduced by a given actuator as a function of applied actuatation parameter can be determined from a set of calibration images captured with differing degrees of defocus.

This project is currently in a partially incomplete state.

(Tested with Python 3.11.3 on the most current version of Arch Linux.)

## Requirements

## Installation

To download the project, clone this repository with
```
git clone https://github.com/husicj/dm-calibration.git
```
To add the calibration script to your path on Unix-like systems, run
```
export PATH=path-to-cloned-repo/dm-calibration:$PATH
```

## Usage

For basic usage, run the command
```
python dm-calibration.py {data directory}
```
This supports the directory structure `{data directory}/{actuator directories}/{multi-channel tiff images}`. Also note that digits in the actuator directory name are used to label the resulting data. Similarly, decimal numbers, potentially with a leading sign, in the image file names are assumed to be the actuator voltages for the data within the file.

The resulting data is output in the file "calibration\_data\_coefficients.py". When this file is imported as a python module, it supplies a dictionary keyed by the actuator directory names, containing dictionaries of coefficients keyed by voltage values, and ordered according to their ANSI indexing. The current default also excludes the first three Zernike coefficients, for the polynomials of order 0 and 1.

To run the calibration program with the provided sample data, run the command
```
python dm-calibration.py
```
without additional arguments.

## Acknowledgements

## License

