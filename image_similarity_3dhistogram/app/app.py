#import
import os
import cv2

from flask import Flask, request, render_template
from pymongo import MongoClient
from datetime import datetime
from imagedescriptor.colordescriptor import ColorDescriptor
from imagedescriptor.searcher import Searcher

#constant definitions
app = Flask(__name__)

INDEX = os.path.join(os.path.dirname(__file__), 'index.csv')

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# connect to mongo database
client = MongoClient(
    os.environ['DB_PORT_27017_TCP_ADDR'],
    27017)

# refer to the databse "hsvhist"
db = client.hsvhist

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
        # upload failed
        return render_template("uploaderror.html")

    try:
        #search for the image similarity - starts
        
        RESULTS_ARRAY = []

        # initialize the image descriptor
        cd = ColorDescriptor((8, 12, 3))

        # load the uploaded image and describe it i.e. extract features
        print os.path.abspath(destination)
        from skimage import io
        query = io.imread(destination)
        print query.shape
        features = cd.describe(query)

        # perform the search
        searcher = Searcher(INDEX)
        results = searcher.search(features)

        # loop over the results, add the score and image name
        for (score, resultID) in results:
            RESULTS_ARRAY.append(
                {"image": str(resultID), "score": str(score)})
            
        print RESULTS_ARRAY
        # success - display top 3 results

        # store the uploaded image filename (full), extracted features, top 3 results and datetime in mongo db
        item_doc = {
        'name': destination,
        'description': str(features),
        'result': str(RESULTS_ARRAY[::-1][:3]),
        'uploaded time': datetime.now()
        }

        #inset record
        db.hsvhist.insert_one(item_doc)

        return render_template("search.html", image_names=(RESULTS_ARRAY[::1][:3]), input=filename)

    except:
        # no results
        return render_template("searcherror.html")


if __name__ == "__main__":
    app.run('0.0.0.0', debug=True)
