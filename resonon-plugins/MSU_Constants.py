# Written by Andrew Donelick
# andrew.donelick@msu.montana.edu
# 27 October 2015
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

CLASSES = ["1", "2", "3"]

# Information for saving or retrieving the trained ML models
MODEL_PATH = None

