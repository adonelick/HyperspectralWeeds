# Written by Andrew Donelick
# andrew.donelick@msu.montana.edu
# 13 April 2016
# Montana State University - Optical Remote Sensing Lab


from spectronon.workbench.plugin import SelectPlugin
from resonon.utils.spec import SpecChoice
from MSU_Constants import *

import os
import numpy as np


class SimpleSpectraExport(SelectPlugin):
    """Gather training data for our machine learning models""" 
    label = "Simple Spectra Export"
    userLevel = 1

    def setup(self):
        """
        Sets up the spectra export process by requesting the lighting
        condition for the data collection, and the date of data collection

        :return: (None)
        """

        self.lighting = SpecChoice(label='Select the illumination used for this datacube',
                                   values=LIGHTING,
                                   defaultValue=LIGHTING[0])

        self.date = SpecChoice(label='Select the data collection date for this datacube',
                               values=COLLECTION_DATES,
                               defaultValue=COLLECTION_DATES[0])

    def action(self):
        """
        This is the action that occurs when the "Simple Spectra Export"
        button is clicked on a user selection in Spectronon. The user
        selection in Spectronon is saved into the specified file.

        :return: (None)
        """

        # Load the dataCube as a numpy array
        dataCube = self.datacube.getArray(asBIP=True)
        name = self.datacube.getName()
        #dataCubeFilename = self.datacube.getFilename()
        pointList = self.pointlist
        numPoints = len(pointList)
        lines, samples, bands = dataCube.shape

        # Determine the base name for the output CSV file,
        # which folder the file will be saved in
        # root, name = os.path.split(dataCubeFilename)
        name = name[0:-4]
        lighting = self.lighting.value
        startingName = name + '_' + lighting

        # Determine the number of files of the same type/subject
        # as the one we are currently working on
        dataDirectory = DATA_DIRECTORIES[self.date.value]
        dataFiles = os.listdir(dataDirectory)
        count = 0
        for f in dataFiles:
            if startingName in f:
                count += 1

        spectraFilename = startingName + '_' + str(count).zfill(4) + ".csv"
        destination = os.path.join(dataDirectory, spectraFilename)

        # Reformat the data into a 2-d matrix, with each row being 
        # a spectrum from the selected points
        outputData = np.zeros((numPoints, bands))
        for index, point in enumerate(pointList):

            spectrum = dataCube[point[1], point[0], :]
            outputData[index,:] = spectrum

        # Save the data in the specified destination
        np.savetxt(destination, outputData, delimiter=',')

