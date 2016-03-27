# Written by Andrew Donelick
# andrew.donelick@msu.montana.edu
# 26 March 2016
# Montana State University - Optical Remote Sensing Lab


# This file contains a number of constants used throughout other custom 
# scipts for the Spectronon software. If any additional constants are needed
# in new scripts, they should be put in here.

# Dates that data were collected
COLLECTION_DATES = ["2015_1211"]

# Store where the data is located
DATA_DIRECTORIES = {"2015_1211" : "W:\\Huntly Greenhouse 12-11-2015\\Spectra CSV Files",
                    "2015_1211_ML" : "W:\\Huntly Greenhouse 12-11-2015\\Machine Learning Data"}

MODEL_DIRECTORIES = {"2015_1211" : "W:\\Huntly Greenhouse 12-11-2015\\Machine Learning Models"}


# Metadata to be saved with any exported datacubes
# Note: the 'samples' and 'lines' metadata are automatically
# included, regardless of what is stated here.
# (This information is used by the save training data plugin, which is not in use at the moment)
METADATA_TAGS = ["bands", "wavelength", "reflectance scale factor"]

# Information for training data collection
# (This information is used by the save training data plugin, which is not in use at the moment)
TRAIN_PROPORTION = 0.8
TRAINING_DATA_PATH = "C:\Users\q45d465\Documents\Research\TrainingData.mem"
TESTING_DATA_PATH = "C:\Users\q45d465\Documents\Research\TestingData.mem"
SAMPLE_COUNTS_PATH = "C:\Users\q45d465\Documents\Research\SampleCounts.pkl"

TEMP_DIRECTORY = "W:\\temp"

SUSCEPTIBLE = 'sus'
GR_RESISTANT = 'gr'
DR_RESISTANT = 'dr'

CLASSES = [SUSCEPTIBLE, GR_RESISTANT, DR_RESISTANT]

# Label-to-index and index-to-label dictionaries for the resistance classes
LABEL_TO_INDEX = {SUSCEPTIBLE : 0,
                  GR_RESISTANT  : 1,
                  DR_RESISTANT  : 2 }

RESISTANCE_STRINGS = {SUSCEPTIBLE : "Susceptible",
                      GR_RESISTANT  : "Glyphosate Resistant",
                      DR_RESISTANT  : "Dicamba Resistant" }

INDEX_TO_LABEL = {0 : SUSCEPTIBLE,
                  1 : GR_RESISTANT,
                  2 : DR_RESISTANT}

LIGHTING = ["Diffuse", "Direct", "Artificial"]
GRID = "Grid"
WATERSHED = "Watershed"
SEGMENTATION = [GRID, WATERSHED]

# Information for saving or retrieving the trained ML models
MODEL_PATH = None

# Spectral bands to be removed when calibrating 
# Note: removal is optional, and specified if wanted at runtime
BANDS_TO_REMOVE = [(814.4, 827.85)]

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

NDVI_THRESHOLD = 0.7

# Constants which define the machine learning models' abbreviations
SVM = "svm"
K_NEAREST_NEIGHBORS = "knn"
DECISION_TREE = "dt"
RANDOM_FOREST = "rf"
EXTREMELY_RANDOMIZED_TREES = "ert"
ADABOOST = "ab"
GRADIENT_BOOSTING = "gb"
NEURAL_NETWORK = "nn"
ALL = "all"

MODEL_TYPE_TO_NAME = {
    SVM : "Support Vector Machine",
    K_NEAREST_NEIGHBORS : "K-Nearest Neighbors", 
    DECISION_TREE : "Decision Tree", 
    RANDOM_FOREST : "Random Forest", 
    EXTREMELY_RANDOMIZED_TREES : "Extra Random Trees", 
    ADABOOST : "Adaboost",
    GRADIENT_BOOSTING : "Gradient Boosting"
}

MODEL_NAME_TO_TYPE = {
    "Support Vector Machine" : SVM,
    "K-Nearest Neighbors" : K_NEAREST_NEIGHBORS,
    "Decision Tree" : DECISION_TREE,
    "Random Forest" : RANDOM_FOREST,
    "Extra Random Trees" : EXTREMELY_RANDOMIZED_TREES,
    "Adaboost" : ADABOOST,
    "Gradient Boost" : GRADIENT_BOOSTING
}