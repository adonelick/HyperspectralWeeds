# Written by Andrew Donelick
# andrew.donelick@msu.montana.edu
# 25 March 2016
# Montana State University - Optical Remote Sensing Lab

"""
This file contains constants which may be useful in other
files, such as the names of the classes being classified,
etc.
"""

import sys
sys.path.append("..")

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from learning import NeuralNetworkClassifier

# Proportion of data to be used in the training set
TRAIN_PROPORTION = 0.7

# Store where the data is located
DATA_DIRECTORIES = {"2015_1211" : "W:\Huntly Greenhouse 12-11-2015\Spectra CSV Files",
                    "2015_1211_ML" : "W:\Huntly Greenhouse 12-11-2015\Machine Learning Data"}

MODEL_DIRECTORIES = {"2015_1211" : "W:\Huntly Greenhouse 12-11-2015\Machine Learning Models"}


TRAINING_DATA_PATH = "TrainingData.mem"
TESTING_DATA_PATH = "TestingData.mem"
SAMPLE_COUNTS_PATH = "SampleCounts.pkl"

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

NUM_WAVELENGTHS = 240

# All 240 wavelengths included in the spectra 
WAVELENGTHS = [394.6, 396.6528, 398.7056, 400.7584, 402.8112, 404.864,
               406.9168, 408.9696, 411.0224, 413.0752, 415.128, 417.1808,
               419.2336, 421.2864, 423.3392, 425.392, 427.4448, 429.4976,
               431.5504, 433.6032, 435.656, 437.7088, 439.7616, 441.8144,
               443.8672, 445.92, 447.9728, 450.0256, 452.0784, 454.1312,
               456.184, 458.2368, 460.2896, 462.3424, 464.3952, 466.448,
               468.5008, 470.5536, 472.6064, 474.6592, 476.712, 478.7648,
               480.8176, 482.8704, 484.9232, 486.976, 489.0288, 491.0816,
               493.1344, 495.1872, 497.24, 499.2928, 501.3456, 503.3984, 
               505.4512, 507.504, 509.5568, 511.6096, 513.6624, 515.7152, 
               517.768, 519.8208, 521.8736, 523.9264, 525.9792, 528.032, 
               530.0848, 532.1376, 534.1904, 536.2432, 538.296, 540.3488, 
               542.4016, 544.4544, 546.5072, 548.56, 550.6128, 552.6656, 
               554.7184, 556.7712, 558.824, 560.8768, 562.9296, 564.9824, 
               567.0352, 569.088, 571.1408, 573.1936, 575.2464, 577.2992, 
               579.352, 581.4048, 583.4576, 585.5104, 587.5632, 589.616, 
               591.6688, 593.7216, 595.7744, 597.8272, 599.88, 601.9328, 
               603.9856, 606.0384, 608.0912, 610.144, 612.1968, 614.2496, 
               616.3024, 618.3552, 620.408, 622.4608, 624.5136, 626.5664, 
               628.6192, 630.672, 632.7248, 634.7776, 636.8304, 638.8832, 
               640.936, 642.9888, 645.0416, 647.0944, 649.1472, 651.2, 
               653.2528, 655.3056, 657.3584, 659.4112, 661.464, 663.5168, 
               665.5696, 667.6224, 669.6752, 671.728, 673.7808, 675.8336, 
               677.8864, 679.9392, 681.992, 684.0448, 686.0976, 688.1504, 
               690.2032, 692.256, 694.3088, 696.3616, 698.4144, 700.4672, 
               702.52, 704.5728, 706.6256, 708.6784, 710.7312, 712.784, 
               714.8368, 716.8896, 718.9424, 720.9952, 723.048, 725.1008,
               727.1536, 729.2064, 731.2592, 733.312, 735.3648, 737.4176, 
               739.4704, 741.5232, 743.576, 745.6288, 747.6816, 749.7344, 
               751.7872, 753.84, 755.8928, 757.9456, 759.9984, 762.0512, 
               764.104, 766.1568, 768.2096, 770.2624, 772.3152, 774.368, 
               776.4208, 778.4736, 780.5264, 782.5792, 784.632, 786.6848, 
               788.7376, 790.7904, 792.8432, 794.896, 796.9488, 799.0016, 
               801.0544, 803.1072, 805.16, 807.2128, 809.2656, 811.3184, 
               813.3712, 815.424, 817.4768, 819.5296, 821.5824, 823.6352, 
               825.688, 827.7408, 829.7936, 831.8464, 833.8992, 835.952, 
               838.0048, 840.0576, 842.1104, 844.1632, 846.216, 848.2688, 
               850.3216, 852.3744, 854.4272, 856.48, 858.5328, 860.5856, 
               862.6384, 864.6912, 866.744, 868.7968, 870.8496, 872.9024, 
               874.9552, 877.008, 879.0608, 881.1136, 883.1664, 885.2192]

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

MODELS = {
    SVM : ("Support Vector Machine", SVC),
    K_NEAREST_NEIGHBORS : ("K-Nearest Neighbors", KNeighborsClassifier),
    DECISION_TREE : ("Decision Tree", DecisionTreeClassifier),
    RANDOM_FOREST : ("Random Forest", RandomForestClassifier),
    EXTREMELY_RANDOMIZED_TREES : ("Extra Random Trees", ExtraTreesClassifier),
    ADABOOST : ("Adaboost", AdaBoostClassifier),
    GRADIENT_BOOSTING : ("Gradient Boost", GradientBoostingClassifier),
    NEURAL_NETWORK : ("Neural Network", NeuralNetworkClassifier)
}


# Hyperparameters used in training the machine learning models
HYPERPARAMETERS = {
    SVM : {},
    K_NEAREST_NEIGHBORS : {},
    DECISION_TREE : {},
    RANDOM_FOREST : {},
    EXTREMELY_RANDOMIZED_TREES : {},
    ADABOOST : {},
    GRADIENT_BOOSTING : {},
    NEURAL_NETWORK : {}
}
