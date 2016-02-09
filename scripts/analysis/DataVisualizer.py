# Written by Andrew Donelick
# andrew.donelick@msu.montana.edu
# 4 November 2015
# Montana State University - Optical Remote Sensing Lab

"""
This class is meant to handle the data visualization code,
including plotting, data dimensionality reduction (for visualization
purposes), and retrieving basic metadata.

"""

import numpy as np


class DataVisualizer:
    """
    A class which contains useful functions for visualizing either unlabeled
    or labeled feature vectors. Options are available for reducing the
    dimensionality of the data, selecting which features to plot, and printing
    metadata.
    """

    def __init__(self, X, y=None):
        """
        Construct a new data visualization object, using the data you wish
        to visualize. Labels for the feature vectors are optional.

        :param X:
        :param y: 

        :return: (DataVisualizer) new data visualization object
        """
        
        self.X = X
        self.y = y

    
    def dataReport(self):
        """
        Creates short printout containing basic information about the 
        data being visualized. The prinout includes how many samples are
        stored, the number of members per class (if applicable).

        :return: (None)
        """

        # Extract the information we want to print
        numSamples = 0

        print "Total number of samples:", numSamples


        print "Number of samples per class:"
    

    def reduceDimensionality(self):
        """

        """
        pass


    def visualize(self, numSamples=-1):
        """

        """
        pass