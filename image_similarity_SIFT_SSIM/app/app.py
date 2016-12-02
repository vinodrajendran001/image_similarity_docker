#App using SIFT and SSIM algorithm

from flask import Flask, request, render_template

import os
import cv2
import numpy as np
import ssim.ssimlib as pyssim
from skimage.measure import structural_similarity as ssim
from pymongo import MongoClient
from datetime import datetime

# Constant definitions
SIM_IMAGE_SIZE = (640, 480)
SIFT_RATIO = 0.7
# Replace 'SIFT' with 'SSIM' (if you want to execute SSIM)
algorithm = 'SIFT'

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# connect to mongo database
client = MongoClient(
    os.environ['DB_PORT_27017_TCP_ADDR'],
    27017)

# refer to the databse "SIFTSSIM"
db = client.siftssim

dir_name = os.path.join(APP_ROOT, 'static/')

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    #upload the image to the server

    try:
        target = os.path.join(APP_ROOT, 'images/')
        print(target)
        if not os.path.isdir(target):
                os.mkdir(target)
        else:
            print("Couldn't create upload directory: {}".format(target))
        print(request.files.getlist("file"))
        for upload in request.files.getlist("file"):
            print(upload)
            print("{} is the file name".format(upload.filename))
            filename = upload.filename
            destination = "/".join([target, filename])
            print ("Accept incoming file:", filename)
            print ("Save it to:", destination)
            upload.save(destination)
    
    except:
        return render_template("uploaderror.html")


    #search for the image similarity section

    try:     
        #array to store results
        RESULTS_ARRAY = []

        images = os.listdir(dir_name)
        num_images = len(images)

        #form similarity matrix
        print("Building the similarity matrix using SIFT algorithm for images")
        sm = np.zeros(shape=(1, num_images), dtype=np.float64)
        np.fill_diagonal(sm, 1.0)

        # compute and find the similarity score for the uploaded image
        k = 0
        for i in range(sm.shape[0]):
            for j in range(sm.shape[1]):
                j = j + k
                if i!=j and j < sm.shape[1]:
                    filenames, descriptors, sm[i][j] = get_image_similarity('%s/%s' % (dir_name, filename),
                                                    '%s/%s' % (dir_name, images[j]),
                                                    algorithm=algorithm)
                RESULTS_ARRAY.append(
                {"image": str(images[j]), "score": str(sm[i][j])})


            k += 1

        #sort the result according to the score i.e. high score means more similarity
        RESULTS_ARRAY = sorted(RESULTS_ARRAY, key=lambda k: k['score'])

        # store the uploaded image filename (full), extracted features, top 3 results and datetime in mongo db
        item_doc = {
        'name': filenames,
        'description': str(descriptors),
        'result': str(RESULTS_ARRAY[::-1][:3]),
        'uploaded time': datetime.now()
        }

        #inset record
        db.siftssim.insert_one(item_doc)

        # success - display top 3 results
        return render_template("search.html", image_names=(RESULTS_ARRAY[::-1][:3]), input=filename)
    
    except:
        return render_template("searcherror.html")

def get_image_similarity(img1, img2, algorithm):

    # Converting to grayscale and resizing
    i1 = cv2.resize(cv2.imread(img1, cv2.IMREAD_GRAYSCALE), SIM_IMAGE_SIZE)
    i2 = cv2.resize(cv2.imread(img2, cv2.IMREAD_GRAYSCALE), SIM_IMAGE_SIZE)

    similarity = 0.0

    if algorithm == 'SIFT':
        #detect and compute the key points from the image
        sift = cv2.xfeatures2d.SIFT_create()
        k1, d1 = sift.detectAndCompute(i1, None)
        k2, d2 = sift.detectAndCompute(i2, None)

        #brute-force matcher with knn match is defined, where k = 2
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(d1, d2, k=2)

        for m, n in matches:
            if m.distance < SIFT_RATIO * n.distance:
                similarity += 1.0

        # Custom normalization for better variance in the similarity matrix
        if similarity == len(matches):
            similarity = 1.0
        elif similarity > 1.0:
            similarity = 1.0 - 1.0/similarity
        elif similarity == 1.0:
            similarity = 0.1
        else:
            similarity = 0.0

    elif algorithm == 'SSIM':
        # Default SSIM implementation of Scikit-Image
        similarity = ssim(i1, i2)

    return img1, d1, similarity

if __name__ == "__main__":
    app.run('0.0.0.0', debug=True)
