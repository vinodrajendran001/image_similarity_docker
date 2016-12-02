import os
import cv2

from flask import Flask, request, render_template, send_from_directory, jsonify

from imagedescriptor.colordescriptor import ColorDescriptor
from imagedescriptor.searcher import Searcher

app = Flask(__name__)

INDEX = os.path.join(os.path.dirname(__file__), 'index.csv')

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
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

    # return send_from_directory("images", filename, as_attachment=True)
    return render_template("uploaderror.html", image_name=filename)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

@app.route('/search')
def search():

    RESULTS_ARRAY = []

    # get url
    image_query = os.listdir('./images')
    for query in image_query:
        input = query

    # initialize the image descriptor
    cd = ColorDescriptor((8, 12, 3))
    # load the query image and describe it

    print(APP_ROOT)
    from skimage import io
    query = io.imread('/home/vinod/PycharmProjects/image_similarity/images/image_1.png')
    print query.shape
    features = cd.describe(query)

    # perform the search
    searcher = Searcher(INDEX)
    results = searcher.search(features)

    # loop over the results, displaying the score and image name
    for (score, resultID) in results:
        RESULTS_ARRAY.append(
            {"image": str(resultID), "score": str(score)})

    return render_template("search.html", image_names=(RESULTS_ARRAY[::-1][:3]))


if __name__ == "__main__":
    app.run(port=4555, debug=True)