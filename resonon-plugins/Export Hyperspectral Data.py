# Written by Andrew Donelick
# andrew.donelick@msu.montana.edu
# 22 October 2015
# Montana State University - Optical Remote Sensing Lab

from spectronon.workbench.plugin import SelectPlugin

import numpy as np
import cPickle
import os
from MSU_Constants import METADATA_TAGS

class HyperspectralExport(SelectPlugin):
    """Export a datacube as a numpy array""" 
    label = "MSU Cube Export"
    userLevel = 1

    def action(self):
        """
        This is the action that occurs when the "MSU Cube Export"
        button is clicked on a user selection in Spectronon.

        :return: (None)
        """

        dataCube = self.datacube.getArray(asBIP=True)
        
        # Extract the coordinates of the corners for the
        # bounding rectangle of the user's selection
        pointList = self.pointlist

        samples = map(lambda x: x[0], pointList)
        lines = map(lambda x: x[1], pointList)

        minLine = min(lines)
        maxLine = max(lines)
        minSample = min(samples)
        maxSample = max(samples)

        # Extract the user's selection from the datacube
        dataCube = dataCube[minLine:maxLine, minSample:maxSample, :]
        lines, samples, bands = dataCube.shape

        # Store any useful metadata
        metadata = {"lines" : lines, "samples": samples}
        for tag in METADATA_TAGS:
            try:
                metadata[tag] = self.datacube._metadata[tag]
            except KeyError:
                metadata[tag] = None

        # Determine the location to save the data to
        destination = self.wb.requestOpenFilename(wildcard="*.npy",
                                                  message="Select a Filename for the Datacube",
                                                  multiple=False)
        if not destination:
            return
        folder, name = os.path.split(destination)

        # Save the metadata in a file seperate from the hyperspectral data
        metadataFilename = name[:-4] + "_Metadata.pkl"
        fileHandle = open(os.path.join(folder, metadataFilename), 'wb')
        cPickle.dump(metadata, fileHandle)
        fileHandle.close()

        # Save the datacube
        np.save(destination, dataCube)
