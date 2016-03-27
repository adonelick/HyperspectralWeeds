# Written by Andrew Donelick
# andrew.donelick@msu.montana.edu
# 4 March 2016
# Montana State University - Optical Remote Sensing Lab

"""
Automatically divides up the visible plant material in a dataCube
and saves each chunk as a "leaf" for later analysis. No need to
draw regions of interest to select the data to export. Note:
this method is not perfect, and can save inconsistently illuminated
regions in a single data file, which can mess up later analysis.
Use with care!
"""

from spectronon.workbench.plugin import SelectPlugin
from resonon.utils.spec import SpecChoice
from resonon.utils.spec import SpecFloat
from resonon.core.data import util

from MSU_Constants import TEMP_DIRECTORY
from MSU_Constants import LIGHTING
from MSU_Constants import SEGMENTATION

import numpy as np
import math
import os
import cv2
from matplotlib import pyplot as plt

OPENCV_VERSION = "3.1.0"

class AutoSpectraExport(SelectPlugin):
    """Calibrate the spectra to reflectance with an internal target""" 
    label = "Auto Spectra Export"
    userLevel = 1

    def setup(self):
        """
        Allows the user to select the type of illumination used in the 
        datacube being viewed and processed.

        :return: (None)
        """

        self.lighting = SpecChoice(label='Select the illumination used for this datacube',
                                   values=LIGHTING,
                                   defaultValue=LIGHTING[0])

        self.extractionMethod = SpecChoice(label='Select the segementation method',
                                           values=SEGMENTATION,
                                           defaultValue=SEGMENTATION[0])

        self.ndviThreshold = SpecFloat(label='NDVI Threshold', 
                                       minval=0, 
                                       maxval=1,
                                       stepsize=0.05, 
                                       defaultValue=0.75)

    def action(self):
        """
        Locates the plant material in an image, divides that plant material up
        into segements and saves each segment into a specified folder.

        :return: (None)
        """

        # Check the version of OpenCV (necessary for watershed segmentation)
        if cv2.__version__ != OPENCV_VERSION:
            self.wb.postMessage("Please update OpenCV to version 3.1.0")
            return

        # Retrieve useful information about the datacube and the user selection
        dataCube = self.datacube.getArray(asBIP=True)
        datacubeName = self.datacube.getName()
        destination = self.wb.requestDirectory(message="Please Give a Destination Folder for the Data")
        if destination == None:
            return

        # Construct an image which shows the location of the plant material in the image
        ndvi = NDVI(self.datacube)
        plantMaterial = np.zeros(ndvi.shape, np.uint8)
        plantLocations = ndvi >= self.ndviThreshold.value
        plantMaterial[plantLocations] = 255

        extractionMethod = self.extractionMethod.value
        lighting = self.lighting.value

        if extractionMethod == GRID:
            gridSegmentation(self.wb, dataCube, plantMaterial, lighting, destination, datacubeName)
        elif extractionMethod == WATERSHED:
            watershedSegmentation(self.wb, dataCube, plantMaterial, lighting, destination, datacubeName)
        else:
            self.wb.postMessage("Unknown data segmentation method!")


def gridSegmentation(wb, dataCube, plantMaterial, lighting, destination, datacubeName):
    """
    Segements the plant material with a simple grid algorithm, 
    saves the segmented data in CSV files.

    :param wb: (Spectronon workbench)
    :param dataCube: (numpy array) BIL Datacube
    :param plantMaterial: (numpy array) Locations of the plant material
    :param lighting: (string) Illumination description string
    :param destination: (string) Directory in which to save the data
    :param datacubeName: (string) Name of the current data cube

    :return: (None)
    """

    lines, samples, bands = dataCube.shape
    
    # Break the image up into segements to save
    numLines = 40
    numSamples = 60
    lineWidth = math.floor(1.0*lines/numLines)
    sampleWidth = math.floor(1.0*samples/numSamples)

    leaf = 0
    for line in xrange(numLines):

        firstLine = line*lineWidth
        lastLine = firstLine + lineWidth

        for sample in xrange(numSamples):

            # Create a mask which will select only the region of the
            # grid we are currently interested in
            firstSample = sample*sampleWidth
            lastSample = firstSample + sampleWidth

            leafMask = np.zeros(plantMaterial.shape, np.uint8)
            leafMask[firstLine:lastLine, firstSample:lastSample] = 255
            leafMask = cv2.bitwise_and(leafMask, plantMaterial)

            # If there are non non-zero pixels in this segement, don't save 
            # the corresponding data file, as it will be empty
            if max(leafMask.flatten()) == 0:
                continue

            # Extract the data to save, save it to the file
            outputData = dataCube[leafMask != 0, :]

            filename = datacubeName + '_' + lighting + '_' + str(leaf).zfill(4) + ".csv"
            fileLocation = os.path.join(destination, filename)
            np.savetxt(fileLocation, outputData, delimiter=',')

            leaf += 1


def watershedSegmentation(wb, dataCube, plantMaterial, lighting, destination, datacubeName):
    """
    Segements the plant material with watershed segmentation algorithm, 
    saves the segmented data in CSV files.

    :param wb: (Spectronon workbench)
    :param dataCube: (numpy array) BIL Datacube
    :param plantMaterial: (numpy array) Locations of the plant material
    :param lighting: (string) Illumination description string
    :param destination: (string) Directory in which to save the data
    :param datacubeName: (string) Name of the current data cube

    :return: (None)
    """
    
    lines, samples, bands = dataCube.shape

    plantMaterialGray = plantMaterial
    plantMaterial = cv2.merge((plantMaterialGray, plantMaterialGray, plantMaterialGray))

    # Remove noise from the image
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(plantMaterialGray, cv2.MORPH_OPEN, kernel)

    # Locate the for-sure background area
    sure_bg = cv2.dilate(opening, kernel)
    
    # Finding the for-sure foreground area
    dist_transform, labels = cv2.distanceTransformWithLabels(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.4*dist_transform.max(), 255, 0)
    sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_CLOSE, kernel)

    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    
    # Now, mark the region of unknown with zero. We will have the segmented "leaves"
    # present in the image
    markers[unknown==255] = 0
    markers = cv2.watershed(plantMaterial, markers)
    numMarkers = max(markers.flatten())

    plt.imshow(markers)
    plt.show()

    saveData = wb.postQuestion("Continue with saving the segmented data?", 
                               title="Segmentation Okay?")
    if not saveData:
        return

    # Save the data for each leaf in a separate csv file (skip the background)
    for leaf in xrange(2, numMarkers):

        outputData = dataCube[markers==leaf, :]
        if len(outputData.flatten()) / 240 < 2:
            break

        filename = datacubeName + '_' + lighting + '_' + str(leaf).zfill(4) + ".csv"
        fileLocation = os.path.join(destination, filename)

        np.savetxt(fileLocation, outputData, delimiter=',')


def NDVI(datacube):
    """
    Calculate the NDVI index for every spectrum in a given datacube

    :param datacube: (Spectronon Datacube) 

    :return: (np.array) NDVI index 
    """
    ir_band = datacube.getBandAtWavelength(800).astype('f')
    red_band = datacube.getBandAtWavelength(680).astype('f')

    result = np.zeros(datacube.getBand(0).shape, dtype='f')
    result[:,:] = -2
    numer = (ir_band - red_band)
    denom = (ir_band + red_band)
    result[(denom)>0] = (numer / denom)[(denom)>0]
   
    return result

