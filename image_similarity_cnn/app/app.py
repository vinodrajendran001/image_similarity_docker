# App using CAE/VAE with cosine similarity

#import

from flask import Flask, request, render_template

import os
import cv2
import numpy as np
from pymongo import MongoClient
from datetime import datetime

import h5py
from sklearn.metrics.pairwise import cosine_similarity
from numpy import *
from keras.models import load_model
from numpy import linalg as LA

#constant definition
app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# connect to mongo database
client = MongoClient(
    os.environ['DB_PORT_27017_TCP_ADDR'],
    27017)

# refer to the databse "caevae"
db = client.caevae

dir_name = os.path.join(APP_ROOT, 'static/')

models = os.path.join(os.path.dirname(__file__), 'model.h5')

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

    model = load_model(models)
    #extract the uploaded img features from model by predicting
    im = cv2.imread(destination)
    im = cv2.resize(im, (28, 28))
    im = array([array(im)])
    out = model.predict(im)
    #normalized
    up_features = out[0]/LA.norm(out[0])


    RESULTS_ARRAY = []

    images = os.listdir(dir_name)


    for i, imgPath in enumerate(images):
        # compute features from db and compare with uploaded img features 
        im = cv2.imread('%s%s' % (dir_name, images[i]))
        im = cv2.resize(im, (28, 28))
        im = array([array(im)])
        out = model.predict(im)
        #Normalize
        out = out[0] / LA.norm(out[0])
        # flatten into 1D and apply cosine similarity
        scores = cosine_similarity(np.array(up_features).flatten().reshape((1, -1)), np.array(out).flatten().reshape((1, -1)))
        RESULTS_ARRAY.append(
                    {"image": str(images[i]), "score": str(scores)})


    RESULTS_ARRAY = sorted(RESULTS_ARRAY, key=lambda k: k['score'])

    # store the uploaded image filename (full), extracted features, top 3 results and datetime in mongo db
    item_doc = {
    'name': destination,
    'description': str(up_features),
    'result': str(RESULTS_ARRAY[::-1][:3]),
    'uploaded time': datetime.now()
    }

    #inset record
    db.caevae.insert_one(item_doc)    

    # success - display top 3 results
    return render_template("search.html", image_names=(RESULTS_ARRAY[::-1][:3]), input=filename)


if __name__ == "__main__":
    app.run('0.0.0.0', debug=True)
