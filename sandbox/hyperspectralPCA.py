# Written by Andrew Donelick
# andrew.donelick@msu.montana.edu
# 26 October 2015
# Montana State University - Optical Remote Sensing Lab


import numpy as np
import os
import cPickle
import mkl
from sklearn.decomposition import IncrementalPCA

import plotly.plotly as plt
import plotly.graph_objs as go


PATH = "C:\\Users\\q45d465\\Documents\\Research\\Test1"
NAME = "test2.npy"


def main():

    mkl.set_num_threads(8)

    # Load the metadata for the datacube
    metadataFilePath = os.path.join(PATH, NAME[:-4] + "_Metadata.pkl")
    metadata = cPickle.load(open(metadataFilePath, 'rb'))

    lines = metadata["lines"]
    samples = metadata["samples"]
    bands = metadata["bands"]
    
    dataCube = np.load(os.path.join(PATH, NAME))

    X = np.zeros((lines*samples, bands))

    index = 0
    for line in xrange(lines):
        for sample in xrange(samples):
            X[index,:] = dataCube[line, sample,:]
            index += 1

    del dataCube
    pca = IncrementalPCA(n_components=3)

    transformedX = pca.fit_transform(X)

    indices = np.random.choice(lines*samples, size=2000, replace=False)
    selectedSamples = transformedX[indices]

    x = selectedSamples[:,0]
    y = selectedSamples[:,1]
    z = selectedSamples[:,2]

    trace = go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers',
    marker=dict(
        size=3,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.25
                 ),
                 opacity=1
        )
    )


    data = [trace]
    fig = go.Figure(data=data)
    plot_url = plt.plot(fig, filename='Hyperspectral PCA Test')



if __name__ == '__main__':
    main()