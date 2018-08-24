import deepmatching as dm
#help(dm.deepmatching)
#dm.deepmatching() # show some help about options
from PIL import Image
import numpy as np
img1 = np.array(Image.open('liberty1.png'))
img2 = np.array(Image.open('liberty2.png'))
matches = dm.deepmatching( img1, img2, '-downscale 2 -v' )
matches
