import numpy as np
import glob
import os
from scipy.ndimage import zoom

EXT = '.dat'

def rescale_write():

    ALL_FILES = glob.glob("../shared/Data/HMI_LOS_SHARPS/valid_magnetograms/*flaring/*/*.dat")

    for i in range(len(ALL_FILES)):
        FILE = ALL_FILES[i]
        IMAGE = np.load(FILE)
        
        (height, width) = IMAGE.shape
        
        """
        if (np.isnan(IMAGE).any()):
            continue
        """
        NEW_IMAGE = zoom(IMAGE, (128.0/height, 128.0/width), mode="wrap")
        assert(NEW_IMAGE.shape == (128, 128))

        NEW_IMAGE_FILENAME = os.path.split(FILE)[-1][0:-4]
        WRITE_PATH = os.path.abspath("../shared/Data/HMI_LOS_SHARPS/valid_magnetograms/los2")

        NAME = os.path.join(WRITE_PATH, NEW_IMAGE_FILENAME)
        NAMEEXT = NAME + EXT
        NEW_IMAGE.dump(NAMEEXT)
        #NEW_RESIZED_IMAGE.dump(NAME)

rescale_write()