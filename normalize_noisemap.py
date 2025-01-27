import nibabel as nib
from skimage import morphology
import numpy as np

def normalize_noisemap(noisemap, data, mask, bvalues):
#Take the estimated noisemaps, erode the brain mask slightly, and divide by the
#mean b0

    kernel = morphology.cube(9)
    mask = morphology.erosion(mask,kernel)

    b0mean = np.nanmean(data[:,:,:,bvalues<100], 3)
    term1 = np.divide(noisemap,b0mean)
    noisemap = np.multiply(term1,mask)

    return noisemap