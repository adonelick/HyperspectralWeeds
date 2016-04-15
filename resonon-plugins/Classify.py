# Written by Andrew Donelick
# andrew.donelick@msu.montana.edu
# 15 April 2016
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

PLANT_MATERIAL = 100
BACKGROUND = -1

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
        pointList = self.pointlist
        lines, samples, bands = dataCube.shape

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

        # Classify the remaining plant material using the classifier
        modelType = MODEL_NAME_TO_TYPE[self.classifier.value]
        clf = loadModel(self.wb, self.date.value, modelType)
        if clf == None:
            return

        classifiedPlantMaterial = np.zeros((lines, samples))
        spectra = []
        points = []

        # Collect the spectra we wish to classify (those which are
        # from the plants, not the background)
        for line in xrange(minLine, maxLine + 1):
            for sample in xrange(minSample, maxSample + 1):

                if (background[line, sample] == PLANT_MATERIAL):
                    spectrum = dataCube[line, sample, :]
                    spectrum = np.nan_to_num(spectrum)

                    spectra.append(spectrum)
                    points.append((line, sample))

        # Using the classifier, label the collected spectra, and build
        # the classification map
        labels = clf.predict(np.array(spectra))
        for i, p in enumerate(points):
            label = labels[i]
            line, sample = p
            classifiedPlantMaterial[line, sample] = label + 1

        # Plot the classification map
        cmap = cm.get_cmap('seismic')
        maxIndex = max(LABEL_TO_INDEX.values()) + 1
        classifiedPlantMaterial[0, 0] = maxIndex

        rgbaBG = cmap(0.0)
        rgbaSUS = cmap(1.0*(LABEL_TO_INDEX[SUSCEPTIBLE]+1) / maxIndex)
        rgbaDR = cmap(1.0*(LABEL_TO_INDEX[DR_RESISTANT]+1) / maxIndex)
        rgbaGR = cmap(1.0*(LABEL_TO_INDEX[GR_RESISTANT]+1) / maxIndex)

        bgPatch = mpatches.Patch(color=rgbaBG, label="Background")
        susPatch = mpatches.Patch(color=rgbaSUS, label=RESISTANCE_STRINGS[SUSCEPTIBLE])
        drPatch = mpatches.Patch(color=rgbaDR, label=RESISTANCE_STRINGS[DR_RESISTANT])
        grPatch = mpatches.Patch(color=rgbaGR, label=RESISTANCE_STRINGS[GR_RESISTANT])

        plt.imshow(classifiedPlantMaterial, cmap=cmap)
        plt.legend(handles=[bgPatch, susPatch, grPatch, drPatch])
        plt.title("Classification Map")
        plt.show()


def loadModel(wb, date, modelType):
    """
    Loads a trained machine learning model from disk for use.

    :param wb: (Spectronon workbench) Reference to useful stuff in Spectronon
    :param date: (string) Date in which the data was collected (YYYY_MMDD)
    :param modelType: (string) Type of model to be loaded (e.g svm, dt, rf, ...)

    :return: (sklearn model) trained machine learning model
    """

    modelDirectory = MODEL_DIRECTORIES[date]
    modelPath = os.path.join(modelDirectory, modelType + ".model")

    try:
        with open(modelPath, 'rb') as fileHandle:
            model = cPickle.load(fileHandle)
            fileHandle.close()

        return model

    except IOError:

        wb.postMessage("Unable to load model of type: " + modelType)
        return None


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
