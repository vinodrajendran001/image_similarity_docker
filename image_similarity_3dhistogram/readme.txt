Prerequistes:

--> Install Docker and Docker-compose
--> mac users remove the .DS_store file present inside the subfolders esp. "static".
--> *"static" folder should contain only images

How to run

step 1:open the terminal

step 2:Run the following command inside "app" folder

	root$ cd image_similarity_3dhistogram/app

	~/image_similarity_3dhistogram/app$ python index.py --dataset dataset --index index.csv

index.csv file will be generated

step 3:Make sure that the "index.csv" file is present inside the app folder i.e image_similarity_3dhistogram/app/

Note: step 2 and 3 needs to be executed only for new set of images.

step 4:In terminal, go to the "image_similarity_3dhistogram" folder run the following commands

	~/image_similarity_3dhistogram/app$ cd ..

	~/image_similarity_3dhistogram$ docker-compose build

	~/image_similarity_3dhistogram$ docker-compose up

Thats it!

step 5:Go to browser and in the url type '0.0.0.0' and press enter

	-file upload page opens
	-choose the image file (.jpg or .png) and click upload

voila...top 3 similar results will be displayed.


Note: Any changes done to "app" folder, then all commands in step 4 has to be executed

--------------------------------------------------------------------------------------------------
To verify storage in database
--------------------------------------------------------------------------------------------------

1. open a new terminal
2. execute the command to open the mongodb shell
	$ docker exec -it imagesimilarity3dhistogram_db_1 mongo
3. switch to "hsvhist" database by executing the following command
	> use hsvhist
4. To view the information stored(filename, features, datetime and top 3 results for uploaded images are displayed)
	> db.hsvhist.find()

