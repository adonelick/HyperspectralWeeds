# Written by Andrew Donelick
# andrew.donelick@msu.montana.edu
# 8 February 2016
# Montana State University - Optical Remote Sensing Lab


# This file contains a number of constants used throughout other custom 
# scipts for the Spectronon software. If any additional constants are needed
# in new scripts, they should be put in here.


# Metadata to be saved with any exported datacubes
# Note: the 'samples' and 'lines' metadata are automatically
# included, regardless of what is stated here.
METADATA_TAGS = ["bands", "wavelength", "reflectance scale factor"]

# Information for training data collection
TRAIN_PROPORTION = 0.8
TRAINING_DATA_PATH = "C:\Users\q45d465\Documents\Research\TrainingData.mem"
TESTING_DATA_PATH = "C:\Users\q45d465\Documents\Research\TestingData.mem"
SAMPLE_COUNTS_PATH = "C:\Users\q45d465\Documents\Research\SampleCounts.pkl"

CLASSES = ["SUS Kochia", "GR Kochia", "DR Kochia"]

# Information for saving or retrieving the trained ML models
MODEL_PATH = None

# Wavelength specific reflectance for the Spectralon calibration panel
# Maps wavelength (nm) to reflectance value
# Source: March 8, 2011 calibration certificate for S/N 7A11D-1394
SPECTRALON_REFLECTANCE = {
    300 : 0.977,
    350 : 0.984,
    400 : 0.989,
    450 : 0.988,
    500 : 0.988,
    550 : 0.989,
    600 : 0.989,
    650 : 0.988,
    700 : 0.988,
    750 : 0.988,
    800 : 0.987,
    850 : 0.987,
    900 : 0.989,
    950 : 0.989,
    1000 : 0.989,
    1050 : 0.988,
    1100 : 0.988,
    1150 : 0.988,
    1200 : 0.987,
    1250 : 0.987,
    1300 : 0.986,
    1350 : 0.985,
    1400 : 0.985,
    1450 : 0.986,
    1500 : 0.986
}

MIN_SPECTRALON_WAVELENGTH = 300
MAX_SPECTRALON_WAVELENGTH = 1500
SPECTRALON_WAVELENGTH_STEP = 50