# Written by Andrew Donelick
# andrew.donelick@msu.montana.edu
# 16 February 2016
# Montana State University - Optical Remote Sensing Lab


from spectronon.workbench.plugin import SelectPlugin
import numpy as np


class SimpleSpectraExport(SelectPlugin):
    """Gather training data for our machine learning models""" 
    label = "Simple Spectra Export"
    userLevel = 1

    def action(self):
        """
        This is the action that occurs when the "Simple Spectra Export"
        button is clicked on a user selection in Spectronon. The user
        selection in Spectronon is saved into the specified file.

        :return: None
        """

        destination = self.wb.requestOpenFilename(message="Please Give a Destination for the Spectra",
                                                  wildcard='*.csv')

        # Load the dataCube as a numpy array
        dataCube = self.datacube.getArray(asBIP=True)
        pointList = self.pointlist
        numPoints = len(pointList)
        lines, samples, bands = dataCube.shape

        # Reformat the data into a 2-d matrix, with each row being 
        # a spectrum from the selected points
        outputData = np.zeros((numPoints, bands))
        for index, point in enumerate(pointList):

            spectrum = dataCube[point[1], point[0], :]
            outputData[index,:] = spectrum

        # Save the data in the specified destination
        np.savetxt(destination, outputData, delimiter=',')

