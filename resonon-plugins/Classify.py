# Written by Andrew Donelick
# andrew.donelick@msu.montana.edu
# 26 March 2016
# Montana State University - Optical Remote Sensing Lab

from spectronon.workbench.plugin import SelectPlugin
from resonon.utils.spec import SpecChoice
from resonon.utils.spec import SpecFloat
from resonon.core.data import util
from resonon.constants import INTERFACES
from MSU_Constants import *

import os
import cPickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches

class ClassifyPlugin(SelectPlugin):
    """
    Classifies the spectra in a datacube using a trained classification
    model created with external data analysis.
    """
    label = "External Classification"
    userLevel = 1

    def setup(self):
        """
        Gather necessary information for classifying spectra in the datacube

        :return: (None)
        """
        self.lighting = SpecChoice(label='Select the illumination used for this datacube',
                                   values=LIGHTING,
                                   defaultValue=LIGHTING[0])

        self.date = SpecChoice(label='Select the data collection date for this datacube',
                               values=COLLECTION_DATES,
                               defaultValue=COLLECTION_DATES[0])

        self.classifier = SpecChoice(label='Select the classifier you wish to use',
                                     values=MODEL_NAME_TO_TYPE.keys(),
                                     defaultValue=MODEL_NAME_TO_TYPE.keys()[0])

        self.ndviThreshold = SpecFloat(label='NDVI Threshold', 
                                       minval=0, 
                                       maxval=1,
                                       stepsize=0.05, 
                                       defaultValue=0.75)


    def action(self):
        """
        Classifies the spectra in a datacube according to an externally trained
        classifier. 

        :return: (None)
        """

        # Load the dataCube as a numpy array
        dataCube = self.datacube.getArray(asBIP=True)
        datacubeName = self.datacube.getName()
        dataCubeFilename = self.datacube.getFilename()
        lines, samples, bands = dataCube.shape

        # Classify the background using an NDVI index. Background is denoted
        # with a value of -1
        ndvi = NDVI(self.datacube)
        background = np.zeros(ndvi.shape, np.int8)
        plantLocations = ndvi >= self.ndviThreshold.value
        background = np.zeros(ndvi.shape, np.int8)
        background[~plantLocations] = -1

        # Classify the remaining plant material using the classifier
        modelType = MODEL_NAME_TO_TYPE[self.classifier.value]
        clf = loadModel(self.date.value, modelType)

        classifiedPlantMaterial = np.zeros((lines, samples))
        for line in xrange(lines):
            for sample in xrange(samples):

                if background[line, sample] == 0:

                    spectrum = dataCube[line, sample, :]
                    label = clf.predict(np.array([spectrum]))
                    classifiedPlantMaterial[line, sample] = label

        totalClassification = background + classifiedPlantMaterial
        totalClassification += 1

        # Plot the classification map
        cmap = cm.get_cmap('seismic')
        maxIndex = max(LABEL_TO_INDEX.values()) + 1
        totalClassification[0, 0] = maxIndex

        rgbaBG = cmap(0.0)
        rgbaSUS = cmap(1.0*(LABEL_TO_INDEX[SUSCEPTIBLE]+1) / maxIndex)
        rgbaDR = cmap(1.0*(LABEL_TO_INDEX[DR_RESISTANT]+1) / maxIndex)
        rgbaGR = cmap(1.0*(LABEL_TO_INDEX[GR_RESISTANT]+1) / maxIndex)

        bgPatch = mpatches.Patch(color=rgbaBG, label="Background")
        susPatch = mpatches.Patch(color=rgbaSUS, label=RESISTANCE_STRINGS[SUSCEPTIBLE])
        drPatch = mpatches.Patch(color=rgbaDR, label=RESISTANCE_STRINGS[DR_RESISTANT])
        grPatch = mpatches.Patch(color=rgbaGR, label=RESISTANCE_STRINGS[GR_RESISTANT])

        plt.imshow(totalClassification, cmap=cmap)
        plt.legend(handles=[bgPatch, susPatch, grPatch, drPatch])
        plt.title("Classification Map")
        plt.show()


def loadModel(date, modelType):
    """
    Loads a trained machine learning model from disk for use.

    :param date: (string) Date in which the data was collected (YYYY_MMDD)
    :param modelType: (string) Type of model to be loaded (e.g svm, dt, rf, ...)

    :return: (sklearn model) trained machine learning model
    """

    modelDirectory = MODEL_DIRECTORIES[date]
    modelPath = os.path.join(modelDirectory, modelType + ".model")

    with open(modelPath, 'rb') as fileHandle:
        model = cPickle.load(fileHandle)
        fileHandle.close()

    return model


def createNewDatacube(workbench, band, wavelength, rotationString):
    """
    Creates a Spectronon DataCube object from the supplied data matrix. 

    :param workbench: (Spectronon workbench) Reference to useful stuff in Spectronon
    :param band: (matrix of floats) Data which should be put into a cube
    :param wavelength: (float) Wavelengths of the datacube
    :param rotationString: (string) Rotation format for the datacube

    :return: (Spectronon DataCube) New datacube containing provided data
    """

    newcube = util.makeEmptyCube(mode="memory",
                                 typechar='f',
                                 rotationString=rotationString)

    newcube.appendBandWithWavelength(band, wave=wavelength)

    return newcube


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
