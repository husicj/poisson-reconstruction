# Deformable Mirror Calibration (v0.3.0)

## Description

The goal of this project is to provide a streamlined process for calibration of a deformable mirror. Specifically, making use of a phase diversity method and a algorithm based on the assumption of Poisson-distributed noise, the wavefront aberrations introduced by a given actuator as a function of applied actuatation parameter can be determined from a set of calibration images captured with differing degrees of defocus.

This project is currently in a partially incomplete state.

(Tested with Python 3.11.3 on the most current version of Arch Linux.)

## Requirements

Dependencies can be found in requirements.txt.

## Installation

To download the project, clone this repository with
```
git clone https://github.com/husicj/dm-calibration.git
```

## Usage

For basic usage, run the command
```
python dm_calibration.py {data directory}
```
This supports the directory structure `{data directory}/{actuator directories}/{multi-channel tiff images}`. Also note that digits in the actuator directory name are used to label the resulting data. Similarly, decimal numbers, potentially with a leading sign, in the image file names are assumed to be the actuator voltages for the data within the file.

When the `CSV_LOG` parameter in dm_calibration.py is set to `True`, the voltage response data (based on a linear fit of the coefficient estimates for each voltage of an actuator) is output in the file "voltage_response.csv" (with units radian/Volt). Each row corresponds to the actuator indicated in column 0, and the remaining columns represent Zernike coefficients, ordered by ANSI index.

Alternatively, when the `LOG_RESULTS` in dm_cailbration.py is set to `True`, the coefficients are output (rather than the slope from linear regression) in the file "calibration\_data\_coefficients.py". When this file is imported as a python module, it supplies a dictionary keyed by the actuator directory names, containing dictionaries of coefficients keyed by voltage values, and ordered according to their ANSI indexing. The current default also excludes the first three Zernike coefficients, for the polynomials of order 0 and 1.

### Cropping

The script currently only supports square images, and its run time increases dramatically with the size of the image. To navigate both of these limitations, you can run
```
python img_crop.py {data directory} {target directory}
```
This assumes the same directory structure as the previous script. Running this automatically crops the images in the data directory around their brightest region (such as the center of a point-like source used for calibration) and saves the results in the target directory with the same directory structure.

Alternatively, you may now simply run the main script `dm_calibration.py` with uncropped images. If the data images are too large, or not square, the script will prompt the user to allow autmatic cropping of the images to be saved in a target directory.

### Sample Data

To run the calibration program with the provided sample data, run the command
```
python dm_calibration.py
```
without additional arguments.

## License

See LICENSE.txt.
