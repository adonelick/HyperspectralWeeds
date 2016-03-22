

import numpy as np 
import skimage.data
import skimage.segmentation
import skimage.morphology
from skimage.feature import peak_local_max
from scipy import ndimage
from skimage.morphology import watershed
from matplotlib import pyplot as plt


def main():
    
    img = skimage.data.imread('leaves.png', as_grey=True)

    distance = ndimage.distance_transform_edt(img)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((25, 25)), labels=img)
    markers = skimage.morphology.label(local_maxi)
    labels_ws = watershed(-distance, markers, mask=img)

    #labels_rw = skimage.segmentation.random_walker(img, markers)

    print max(labels_ws.flatten())
    plt.imshow(labels_ws)
    plt.show()




if __name__ == '__main__':
    main()

