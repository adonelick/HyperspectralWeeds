# HyperspectralWeeds

Developed by Andrew Donelick at Montana State University, Optical Remote Sensing Lab

## Summary
This repository contains code to detect herbicide resistance among Kochia
(and possibly other weeds) using hyperspectral imager data. Included in this 
repository are stand-alone Python scripts, as well as plugins for Spectronon, 
a program written by Resonon, Inc.

## Analysis Pipeline
To analyze herbicide resistance in weeds using a hyperspectral imaging system,
we used the following pipeline for collecting and examining the data:

1. Collect hyperspectral image data of plants from each of the resistance
classes of interest. These hyperspectral data cubes should contain plant
leaves imaged under uniform, consistent lighting conditions, with a 
Spectralon (or equivalent) calibration target.

2. Calibrate the collected data cubes using the Spectronon software. This
calibration step converts the raw intensity data to reflectance, thus removing
any spectral effects of the illumination source. 

3. Extract relevant data from the calibrated data cubes. Most often, this 
relevant data consists of spectra of the plant material in the data cube.

4. Visualize the data to get a sense of the separability between the herbicide
resistance/susceptibility classes.

5. Train machine learning models on the extracted data. These models can be
later applied to data cubes of plants of unknown resistance for testing.

6. Test the system's performance by classifying new data cubes with the
machine learning models. 

## Dependencies
This code depends on following software and packages:

* Python 2.7.11 (Anaconda distribution)
* Plotly 1.8.8
* OpenCV 3.1.0
* Spectronon 2.79 (developed by Resonon, Inc)
