# Written by Andrew Donelick
# andrew.donelick@msu.montana.edu
# 22 October 2015
# Montana State University - Optical Remote Sensing Lab

from spectronon.workbench.plugin import SelectPlugin
from resonon.utils.spec import SpecChoice

import numpy as np
import os
import cPickle
import MSU_Constants

class SaveTrainingData(SelectPlugin):
    """Gather training data for our machine learning models""" 
    label = "Save Training Data"
    userLevel = 1

    def setup(self):
        """
        Determines what the label should be for the selected training data.

        :return: (None)
        """

        self.dataLabel = SpecChoice(label='Select a Label for the Training Data',
                                    values=MSU_Constants.CLASSES,
                                    defaultValue=MSU_Constants.CLASSES[0])
        self.dataLabel.help = 'This dialog allows the user to select the label \
                                for the training data being collected.'


    def action(self):
        """
        This is the action that occurs when the "Save Training Data"
        button is clicked on a user selection in Spectronon. The user
        selection in Spectronon is randomly divided up into training and
        testing components, which are saved in their respective files.

        :return: None
        """

        # Load the dataCube as a numpy array
        dataCube = self.datacube.getArray(asBIP=True)
        pointList = self.pointlist
        numPoints = len(pointList)
        lines, samples, bands = dataCube.shape

        # The last column of the data matrix is the label
        newTrainingData = np.zeros((1, bands + 1))
        newTestingData = np.zeros((1, bands + 1))

        numLabel = MSU_Constants.CLASSES.index(self.dataLabel.value)

        # Determine which points belong to the training set, and
        # which points belong to the testing set.
        for point in pointList:

            spectrum = dataCube[point[1], point[0], :]
            spectrum = np.append(spectrum, numLabel)

            if np.random.random() < MSU_Constants.TRAIN_PROPORTION:
                # Add to the training set
                newTrainingData = np.append(newTrainingData, [spectrum], axis=0)
            else:
                # Add to the testing set
                newTestingData = np.append(newTestingData, [spectrum], axis=0)

        # Remove the first row of zeroes from the training and testing data
        newTrainingData = newTrainingData[1:, :]
        newTestingData = newTestingData[1:, :]

        sampleCounts = self.loadSampleCounts()
        newTrainingSamples, _ = newTrainingData.shape
        newTestingSamples, _ = newTestingData.shape

        trainingSamples = sampleCounts["training"]
        testingSamples = sampleCounts["testing"]

        # Save the training and testing files with the new data
        try: 
            # Open the file for appending
            trainingData = np.memmap(MSU_Constants.TRAINING_DATA_PATH, 
                                     mode='r+',
                                     dtype=np.float32,
                                     shape=(trainingSamples, bands+1))
            testingData = np.memmap(MSU_Constants.TESTING_DATA_PATH, 
                                    mode='r+',
                                    dtype=np.float32, 
                                    shape=(testingSamples, bands+1))

            trainingData = trainingData.copy()
            testingData = testingData.copy()

            trainingData.resize(trainingSamples + newTrainingSamples, bands+1)
            testingData.resize(testingSamples + newTestingSamples, bands+1)

            trainingData[trainingSamples:, :] = newTrainingData
            testingData[testingSamples:, :] = newTestingData

        except IOError:
            # Open the file for the first time to write
            trainingData = np.memmap(MSU_Constants.TRAINING_DATA_PATH, 
                                     mode='w+',
                                     dtype=np.float32, 
                                     shape=newTrainingData.shape)
            testingData = np.memmap(MSU_Constants.TESTING_DATA_PATH, 
                                    mode='w+',
                                    dtype=np.float32,
                                    shape=newTestingData.shape)

            trainingData[:,:] = newTrainingData[:,:]
            testingData[:,:] = newTestingData[:,:]

        # Update the sample counts file
        sampleCounts[self.dataLabel.value+"_training"] += newTrainingSamples
        sampleCounts[self.dataLabel.value+"_testing"] += newTestingSamples
        sampleCounts["training"] += newTrainingSamples
        sampleCounts["testing"] += newTestingSamples
        sampleCounts["bands"] = bands
        self.updateSampleCounts(sampleCounts)

        # Flush the data to disk and close the memmaps
        del trainingData
        del testingData


    def loadSampleCounts(self):
        """
        Determine the number of training and testing samples which have already
        been saved to disk. This function looks into the pickle file
        which stores this information.

        :return: (dictionary) dictionary of the number of samples in each class
                              (for both training and testing), as well as the
                              total number of training and testing samples.
        """

        if not os.path.exists(MSU_Constants.SAMPLE_COUNTS_PATH):
            # If the sample counts file does not exist, then 
            # create on with all the class names currently in the 
            # constants file.

            sampleCounts = {}
            for className in MSU_Constants.CLASSES:
                sampleCounts[className+"_training"] = 0
                sampleCounts[className+"_testing"] = 0

            sampleCounts["training"] = 0
            sampleCounts["testing"] = 0

        else:

            sampleCounts = cPickle.load(open(MSU_Constants.SAMPLE_COUNTS_PATH, 'rb'))

            # Verify that we have not added additional classes to the possible
            # set of classes since sampleCounts was last updated. If so, add
            # the new classes to the sampleCounts file 
            expectedKeys = ["training", "testing"]
            for className in MSU_Constants.CLASSES:
                expectedKeys.append(className+"_training")
                expectedKeys.append(className+"_testing")

            actualKeys = sampleCounts.keys()
            if actualKeys != expectedKeys:

                for key in expectedKeys:
                    if key not in actualKeys:
                        sampleCounts[key] = 0

        return sampleCounts


    def updateSampleCounts(self, sampleCounts):
        """
        Updates the number of training and testing samples recorded
        in the pickle file which stores such things.

        :param sampleCounts: (dictionary) dictionary of the number of samples
                                          in each class (for both training 
                                          and testing), as well as the total 
                                          number of training and testing samples.

        :return: (None)
        """

        cPickle.dump(sampleCounts, open(MSU_Constants.SAMPLE_COUNTS_PATH, 'wb'))










