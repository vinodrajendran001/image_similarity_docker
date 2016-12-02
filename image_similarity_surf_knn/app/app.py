# App using KNN with SURF

from flask import Flask, request, render_template
from pymongo import MongoClient
from datetime import datetime
import os
import cv2
import numpy as np

#constant definitions

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# connect to mongo database
client = MongoClient(
    os.environ['DB_PORT_27017_TCP_ADDR'],
    27017)

# refer to the databse "surfknn"
db = client.surfknn

dir_name = os.path.join(APP_ROOT, 'static/')


@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload():
    # upload the image to the server


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

    # search for the image similarity section


    RESULTS_ARRAY = []

    images = os.listdir(dir_name)

    for i, imgPath in enumerate(images):

        # similarity measure using KNN

        filenames, descripts, sm = knn('%s%s' % (dir_name, filename), '%s%s' % (dir_name, images[i]))

        # problem with the sorted as it is considering only first digit
        # therefore divide the score to 100
        sm = sm / (100 * 1.0)

        RESULTS_ARRAY.append(
            {"image": str(images[i]), "score": str(sm)})

    RESULTS_ARRAY = sorted(RESULTS_ARRAY, key=lambda k: k['score'])
    print RESULTS_ARRAY

    # store the uploaded image filename (full), extracted features, top 3 results and datetime in mongo db
    item_doc = {
    'name': filenames,
    'description': str(descripts),
    'result': str(RESULTS_ARRAY[::-1][:3]),
    'uploaded time': datetime.now()
    }

    #inset record
    db.surfknn.insert_one(item_doc)

    # success - display top 3 results
    return render_template("search.html", image_names=(RESULTS_ARRAY[::-1][:3]), input=filename)


def knn(img1, img2):

    i1 = cv2.cvtColor(cv2.imread(img1),cv2.COLOR_BGR2GRAY)
    i2 = cv2.cvtColor(cv2.imread(img2),cv2.COLOR_BGR2GRAY)

    # SURF key point extraction for input image
    surf = cv2.xfeatures2d.SURF_create(128)
    kp = surf.detect(i1)
    kp, descriptors = surf.compute(i1, kp)

    # Setting up samples and  responses for kNN
    samples = np.array(descriptors)
    responses = np.arange(len(kp), dtype=np.float32)

    # kNN training
    knn = cv2.ml.KNearest_create()
    knn.train(samples, cv2.ml.ROW_SAMPLE, responses)

    # loading a template image i.e rest of the image and searching for similar key points

    keys = surf.detect(i2)
    keys, desc = surf.compute(i2, keys)
    count = 0
    lent = len(desc)

    # Using knn finding the nearest points
    for h, des in enumerate(desc):

        des = np.array(des, np.float32).reshape(1, len(des))
        retval, results, neigh_resp, dists = knn.findNearest(des, 1)
        res, dist = int(results[0][0]), dists[0][0]
        if dist<0.1:
            count += 1

        similarity = count*100/lent

    return img1, descriptors, similarity

if __name__ == "__main__":
    app.run('0.0.0.0', debug=True)
