# Written by Andrew Donelick
# andrew.donelick@msu.montana.edu
# 29 March 2016
# Montana State University - Optical Remote Sensing Lab

from spectronon.workbench.plugin import SelectPlugin
from resonon.utils.spec import SpecChoice
from resonon.utils.spec import SpecFloat
from resonon.core.data import util
from resonon.constants import INTERFACES
from MSU_Constants import *

import os
import math
import cPickle
import numpy as np
from matplotlib import pyplot as plt
from joblib import Parallel, delayed

PLANT_MATERIAL = 100
BACKGROUND = -1

class ShadowRemovalPlugin(SelectPlugin):
    """
    Corrects the shadowed spectra of plant material in a dataCube
    to a spectra which is 
    """
    label = "Shadow Removal"
    userLevel = 1

    def setup(self):
        """
        Gather necessary information for correcting shadowed spectra in the datacube

        :return: (None)
        """
        self.lighting = SpecChoice(label='Select the illumination used for this datacube',
                                   values=LIGHTING,
                                   defaultValue=LIGHTING[0])

        self.date = SpecChoice(label='Select the data collection date for this datacube',
                               values=COLLECTION_DATES,
                               defaultValue=COLLECTION_DATES[0])

        self.ndviThreshold = SpecFloat(label='NDVI Threshold', 
                                       minval=0, 
                                       maxval=1,
                                       stepsize=0.05, 
                                       defaultValue=0.75)


    def action(self):
        """
        Corrects the shadowed plant spectra in a data cube.

        :return: (None)
        """

        # Load the dataCube as a numpy array
        dataCube = self.datacube.getArray(asBIP=True)
        datacubeName = self.datacube.getName()
        pointList = self.pointlist
        lines, samples, bands = dataCube.shape
        wavelengths = self.datacube.getWavelengthList()

        # Extract the region of interest from the selected points
        pointLines = map(lambda x: x[1], pointList)
        pointSamples = map(lambda x: x[0], pointList)

        minLine = min(pointLines)
        maxLine = max(pointLines)
        minSample = min(pointSamples)
        maxSample = max(pointSamples)

        # Classify the background using an NDVI index. Note any
        # locations of plant material for classification
        ndvi = NDVI(self.datacube)
        background = PLANT_MATERIAL * np.ones(ndvi.shape, np.int8)
        plantLocations = ndvi >= self.ndviThreshold.value
        background[~plantLocations] = BACKGROUND
        
        # Calculate the mean spectrum of the selection
        data = dataCube[pointList[:, 1], pointList[:, 0], :]
        mean = np.mean(data, axis=0).astype(self.datacube.getTypeChar()).flatten()

        hypersphericalMean = cartesianToHyperspherical(mean)
        desiredR = hypersphericalMean[0]

        newDataCube = np.zeros_like(dataCube)
        for line in xrange(lines):
            for sample in xrange(samples):

                spectrum = dataCube[line, sample, :]
                hypersphericalSpectrum = cartesianToHyperspherical(spectrum)
                correction = 1.0 * desiredR / hypersphericalSpectrum[0]
                hypersphericalSpectrum[0] *= correction

                newDataCube[line, sample, :] = hypersphericalToCartesian(hypersphericalSpectrum)

        calibratedDatacube = createNewDatacube(self.wb, newDataCube,
                                               wavelengths,
                                               self.datacube.getRotationString())

        # Place the new, calibrated datacube on the workbench for saving,
        # manipulation, or whatever you want to do with it
        newName = datacubeName + "_Corrected"
        self.wb.addCube(calibratedDatacube, name=newName)


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


def cartesianToHyperspherical(x):
    """
    Converts a given vector x (of any length) into hyperspherical coordinates.

    :param x: (np.array) Vector in cartesian coordinates 

    :return: (np.array) Vector x in hyperspherical coordinates
    """
    R = np.linalg.norm(x)
    length = len(x)

    hyperspherical_x = np.zeros(length, np.float32)
    hyperspherical_x[0] = R
    for i in xrange(length-1):

        phi = math.acos(1.0*x[i] / np.linalg.norm(x[i:]))

        if (i == length - 1) and (x[i] < 0):
            phi = 2*math.pi - phi

        hyperspherical_x[i+1] = phi
    
    return hyperspherical_x


def hypersphericalToCartesian(x):
    """
    Converts a given vector x (of any length) into cartesian coordinates.

    :param x: (np.array) Vector in hyperspherical coordinates 

    :return: (np.array) Vector x in cartesian coordinates
    """
    R = x[0]
    length = len(x)

    cartesian_x = np.zeros(length, np.float32)

    for index in xrange(length):

        if index == 0:
            cartesian_x[index] = R*math.cos(x[1])
        elif index == length - 1:
            part = map(lambda y: math.sin(y), x[1:])
            cartesian_x[index] = R*reduce(lambda x, y: x*y, part)
        else:
            part = map(lambda y: math.sin(y), x[1:index + 1])
            cartesian_x[index] = R*reduce(lambda x, y: x*y, part) * math.cos(x[index+1])

    return cartesian_x


def createNewDatacube(workbench, data, wavelengths, rotationString):
    """
    Creates a Spectronon DataCube object from the supplied data matrix. 

    :param workbench: (Spectronon workbench) Reference to useful stuff in Spectronon
    :param data: (matrix of floats) Data which should be put into a cube
    :param wavelengths: (list of floats) Wavelengths of the datacube
    :param rotationString: (string) Rotation format for the datacube

    :return: (Spectronon DataCube) New datacube containing provided data
    """

    newcube = util.makeEmptyCube(mode="memory",
                                 typechar='f',
                                 rotationString=rotationString)

    for i, wavelength in enumerate(wavelengths):

        band = data[:,:, i]
        newcube.appendBandWithWavelength(band, wave=wavelength)

    return newcube
