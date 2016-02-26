# Written by Andrew Donelick
# andrew.donelick@msu.montana.edu
# 4 November 2015
# Montana State University - Optical Remote Sensing Lab

"""
This file contains constants which may be useful in other
files, such as the names of the classes being classified,
etc.
"""

# Store where the data is located
DATA_PATHS = {"2015_1211" : "W:\Huntly Greenhouse 12-11-2015\Spectra CSV Files"}


# Label-to-index and index-to-label dictionaries for the resistance classes
LABEL_TO_INDEX = {'SUS' : 0,
                  'GR'  : 1,
                  'DR'  : 2 }

INDEX_TO_LABEL = {0 : 'SUS',
                  1 : 'GR',
                  2 : 'DR'}
