from PIL import Image
from PIL import ImageOps
image = Image.open('./images_3mm/gamma1.jpg')
inverted_image = ImageOps.invert(image)
inverted_image.save('inverted_gamma.jpg')

