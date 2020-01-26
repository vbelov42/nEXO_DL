import numpy as np
from PIL import Image

img_as_img = Image.open('images_3mm/signal1.jpg')
npimg = np.array(img_as_img)
