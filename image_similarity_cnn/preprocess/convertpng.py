import os
from PIL import Image

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

dir_name = os.path.join(APP_ROOT, 'dataset/')

imgList = [os.path.join(dir_name, f) for f in os.listdir(dir_name) if f.endswith('.png')]
RESULTS_ARRAY = []
for i, imgPath in enumerate(imgList):
    imgName = os.path.split(imgPath)[1]
    print imgName
    im = Image.open(imgList[i])
    im.save("/home/vinod/PycharmProjects/image_upload/input/"+imgName[:-3]+"jpg")